//! Accepting and serving client connections, with caps and a header deadline.
//!
//! The admission semaphore bounds the requests the gateway works on, but a request
//! waiting for admission still holds its connection, and so does an idle keep-alive
//! connection. The listener keeps at most `max` connections open, and at most
//! `per_peer` from one address, so one client cannot take every slot; the next client
//! waits in the kernel's accept queue (or, over its own address's cap, is closed).
//! [`serve`] gives every connection a deadline for the request header: a client that
//! connects and sends nothing, sends its header slowly, or keeps an idle keep-alive
//! connection past the deadline is closed, and its slot goes back. A connection whose
//! writes make no progress for the write deadline (a client that stopped reading a
//! streamed reply, so the kernel's send buffer stays full) is closed too: the response
//! body is dropped with it, and so is the admission permit it held.

use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Router;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use pin_project_lite::pin_project;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Sleep;

/// A `TcpListener` that keeps at most `max` accepted connections open, and at most
/// `per_peer` from one IP address.
#[derive(Debug)]
pub struct LimitedListener {
    inner: TcpListener,
    slots: Arc<Semaphore>,
    per_peer: usize,
    peers: Arc<Mutex<HashMap<IpAddr, usize>>>,
}

impl LimitedListener {
    /// Wrap `inner`; both caps must be greater than zero.
    pub fn new(inner: TcpListener, max: usize, per_peer: usize) -> Self {
        assert!(max > 0, "the connection cap must be greater than zero");
        assert!(
            per_peer > 0,
            "the per-address cap must be greater than zero"
        );
        Self {
            inner,
            slots: Arc::new(Semaphore::new(max)),
            per_peer,
            peers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// The bound address.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Accept the next connection within both caps.
    pub async fn accept(&mut self) -> (LimitedStream, SocketAddr) {
        loop {
            // The semaphore is never closed, so `acquire_owned` cannot fail.
            let slot = Arc::clone(&self.slots)
                .acquire_owned()
                .await
                .expect("the connection semaphore is never closed");
            let (stream, address) = axum::serve::Listener::accept(&mut self.inner).await;
            let peer = address.ip().to_canonical();
            let admitted = {
                let mut peers = self.peers.lock().expect("peer table poisoned");
                let open = peers.entry(peer).or_insert(0);
                let admitted = *open < self.per_peer;
                if admitted {
                    *open += 1;
                }
                admitted
            };
            if !admitted {
                // Over this address's cap: close it at once and give the slot back, then let
                // other tasks run, so a queue full of one address's connections cannot keep
                // this loop busy.
                drop(stream);
                drop(slot);
                tokio::task::yield_now().await;
                continue;
            }
            let stream = LimitedStream {
                inner: stream,
                write_timeout: None,
                stalled: None,
                _slot: slot,
                _peer: PeerSlot {
                    peer,
                    peers: Arc::clone(&self.peers),
                },
            };
            return (stream, address);
        }
    }
}

/// One open connection counted against its address's cap until it is dropped.
#[derive(Debug)]
struct PeerSlot {
    peer: IpAddr,
    peers: Arc<Mutex<HashMap<IpAddr, usize>>>,
}

impl Drop for PeerSlot {
    fn drop(&mut self) {
        let mut peers = self.peers.lock().expect("peer table poisoned");
        if let Some(open) = peers.get_mut(&self.peer) {
            *open -= 1;
            if *open == 0 {
                peers.remove(&self.peer);
            }
        }
    }
}

pin_project! {
    /// An accepted connection that holds its slots until it is dropped.
    #[derive(Debug)]
    pub struct LimitedStream {
        #[pin]
        inner: TcpStream,
        // How long a write may stay blocked before the connection is given up; `None` waits.
        write_timeout: Option<Duration>,
        // Running while a write is blocked; cleared by any write that goes through.
        stalled: Option<Pin<Box<Sleep>>>,
        _slot: OwnedSemaphorePermit,
        _peer: PeerSlot,
    }
}

/// Pass a write's result through, or fail it once writes have been blocked for `limit`.
///
/// A blocked write (`Pending`) starts the clock and arms its timer, so the task is woken
/// at the deadline even if the socket never becomes writable; a write that goes through
/// stops it. The error ends hyper's connection, which drops the response body and with it
/// the admission permit.
fn within_write_deadline<T>(
    stalled: &mut Option<Pin<Box<Sleep>>>,
    limit: Option<Duration>,
    context: &mut Context<'_>,
    poll: Poll<io::Result<T>>,
) -> Poll<io::Result<T>> {
    match poll {
        Poll::Ready(result) => {
            *stalled = None;
            Poll::Ready(result)
        }
        Poll::Pending => {
            let Some(limit) = limit else {
                return Poll::Pending;
            };
            let timer = stalled.get_or_insert_with(|| Box::pin(tokio::time::sleep(limit)));
            match timer.as_mut().poll(context) {
                Poll::Ready(()) => Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "the client has not read the reply for the write timeout",
                ))),
                Poll::Pending => Poll::Pending,
            }
        }
    }
}

impl AsyncRead for LimitedStream {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        self.project().inner.poll_read(context, buffer)
    }
}

impl AsyncWrite for LimitedStream {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.project();
        let poll = this.inner.poll_write(context, buffer);
        within_write_deadline(this.stalled, *this.write_timeout, context, poll)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffers: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.project();
        let poll = this.inner.poll_write_vectored(context, buffers);
        within_write_deadline(this.stalled, *this.write_timeout, context, poll)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.project().inner.poll_flush(context)
    }

    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.project().inner.poll_shutdown(context)
    }
}

/// Serve `router` over HTTP/1.1 on `listener` until the process ends.
///
/// Each connection gets `header_timeout` to send a complete request header, counted
/// from the moment it waits for one: on a new connection and between keep-alive
/// requests. A response that is still streaming (SSE) is not cut by it. A connection
/// whose writes stay blocked for `write_timeout` (the client stopped reading) is closed.
pub async fn serve(
    mut listener: LimitedListener,
    router: Router,
    header_timeout: Duration,
    write_timeout: Duration,
) -> io::Result<()> {
    loop {
        let (mut stream, _) = listener.accept().await;
        stream.write_timeout = Some(write_timeout);
        let _ = stream.inner.set_nodelay(true);
        let service = TowerToHyperService::new(router.clone());
        tokio::spawn(async move {
            let connection = hyper::server::conn::http1::Builder::new()
                .timer(TokioTimer::new())
                .header_read_timeout(header_timeout)
                .serve_connection(TokioIo::new(stream), service)
                .with_upgrades();
            // A closed or timed-out connection is the client's business, not an error here.
            let _ = connection.await;
        });
    }
}
