//! A TCP listener with a cap on open connections.
//!
//! The admission semaphore bounds the requests the gateway works on, but a request
//! waiting for admission still holds its connection, and so does an idle keep-alive
//! connection. This listener takes a slot before each `accept` and the connection
//! gives it back when it is dropped, so at most `max` connections are open at once;
//! the next client waits in the kernel's accept queue instead of costing memory.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use pin_project_lite::pin_project;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// A `TcpListener` that keeps at most `max` accepted connections open.
#[derive(Debug)]
pub struct LimitedListener {
    inner: TcpListener,
    slots: Arc<Semaphore>,
}

impl LimitedListener {
    /// Wrap `inner`; `max` must be greater than zero.
    pub fn new(inner: TcpListener, max: usize) -> Self {
        assert!(max > 0, "the connection cap must be greater than zero");
        Self {
            inner,
            slots: Arc::new(Semaphore::new(max)),
        }
    }
}

impl axum::serve::Listener for LimitedListener {
    type Io = LimitedStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        // The semaphore is never closed, so `acquire_owned` cannot fail.
        let slot = Arc::clone(&self.slots)
            .acquire_owned()
            .await
            .expect("the connection semaphore is never closed");
        let (stream, address) = axum::serve::Listener::accept(&mut self.inner).await;
        (
            LimitedStream {
                inner: stream,
                _slot: slot,
            },
            address,
        )
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

pin_project! {
    /// An accepted connection that holds its slot until it is dropped.
    #[derive(Debug)]
    pub struct LimitedStream {
        #[pin]
        inner: TcpStream,
        _slot: OwnedSemaphorePermit,
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
        self.project().inner.poll_write(context, buffer)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffers: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        self.project().inner.poll_write_vectored(context, buffers)
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
