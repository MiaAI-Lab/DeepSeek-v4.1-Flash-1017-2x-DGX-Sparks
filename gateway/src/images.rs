//! Image resolution for the API path: fetch, validate, count tokens, and build the
//! engine's `content_parts` in placeholder order.
//!
//! No OpenCV: the `image` crate reads only the header dimensions, the V4.1 token
//! specification counts the tokens, and the original bytes go to the engine as
//! `data:application/octet-stream;base64,...`.

use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::io::Cursor;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use deepseek_recipe_core::multimodal::{ImageInfo, ImageSource, ImageTokenSpec};
use deepseek_recipe_image::{
    DestinationFilter, ImageByteBudget, ImageError, ImageFetcher, ImageLimits, ImagePreprocessor,
    ImageQuota, ImageResolver, PreprocessOptions, ReqwestImageFetcher,
};
use reqwest::dns::{Name, Resolve, Resolving};
use serde_json::{Value, json};

use crate::ApiError;

/// The image policy of one gateway.
#[derive(Debug, Clone, Default)]
pub struct ImageSettings {
    pub max_images: usize,
    pub max_total_bytes: usize,
    pub block_private_urls: bool,
    /// Test-only name-to-address overrides, applied when building the fetcher.
    pub dns_overrides: HashMap<String, IpAddr>,
}

/// The images of one request after resolution.
#[derive(Debug, Clone, Default)]
pub struct ResolvedImages {
    /// Extra input tokens over one placeholder per image.
    pub adjustment: usize,
    /// `content_parts` in marker order, or `None` when there are no images.
    pub parts: Option<Vec<Value>>,
    pub count: usize,
}

/// Resolve every image source of a request, enforcing the API's limits.
pub async fn resolve(
    sources: &[ImageSource],
    settings: &ImageSettings,
) -> Result<ResolvedImages, ApiError> {
    if sources.is_empty() {
        return Ok(ResolvedImages::default());
    }
    let max_image_bytes = (32 * 1024 * 1024).min(settings.max_total_bytes);
    let limits = ImageLimits {
        max_images: settings.max_images,
        max_image_bytes,
        max_total_bytes: settings.max_total_bytes,
        ..ImageLimits::default()
    };
    let fetcher = build_fetcher(settings, max_image_bytes)?;
    let resolver = ImageResolver::new(fetcher, HeaderPreprocessor).with_limits(limits);
    let mut quota = ImageQuota::new();
    let data = resolver
        .resolve(sources, &mut quota)
        .await
        .map_err(image_error)?;
    let adjustment = data
        .image_token_adjustment()
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let parts = data
        .images
        .iter()
        .map(|image| {
            json!({
                "type": "image_url",
                "url": format!(
                    "data:application/octet-stream;base64,{}",
                    base64::engine::general_purpose::STANDARD.encode(&image.data)
                )
            })
        })
        .collect::<Vec<_>>();
    Ok(ResolvedImages {
        adjustment,
        parts: Some(parts),
        count: data.images.len(),
    })
}

/// The fetcher of one resolve call.  Both variants use a client this module builds,
/// so the destination policy is the gateway's own code and the tests exercise the
/// production path; the third-party crate only streams the body under the limits.
enum ImageFetcherImpl {
    Plain(ReqwestImageFetcher),
    Filtered(FilteredFetcher),
}

impl ImageFetcher for ImageFetcherImpl {
    async fn fetch(&self, url: &str, budget: &ImageByteBudget) -> Result<Vec<u8>, ImageError> {
        match self {
            Self::Plain(fetcher) => fetcher.fetch(url, budget).await,
            Self::Filtered(fetcher) => fetcher.fetch(url, budget).await,
        }
    }
}

/// The most redirects one image fetch follows.
const MAX_REDIRECTS: usize = 5;

fn build_fetcher(
    settings: &ImageSettings,
    max_image_bytes: usize,
) -> Result<ImageFetcherImpl, ApiError> {
    let filter = settings
        .block_private_urls
        .then(|| Arc::new(destination_allowed) as Arc<dyn DestinationFilter>);
    build_fetcher_with(filter, &settings.dns_overrides, max_image_bytes)
}

/// Build the fetcher for `filter` (`None`: no destination policy).
///
/// With a filter, every destination is checked by this module: the host of the
/// initial URL when it is an address, every address a host name resolves to (the
/// client connects only to the addresses the resolver returned, so a second DNS
/// answer cannot swap in another one), and every redirect target, which goes
/// through the same two checks.  Proxies from the environment are ignored, since a
/// proxy would make the connection on the client's behalf to an unchecked address.
fn build_fetcher_with(
    filter: Option<Arc<dyn DestinationFilter>>,
    dns_overrides: &HashMap<String, IpAddr>,
    max_image_bytes: usize,
) -> Result<ImageFetcherImpl, ApiError> {
    let overrides: HashMap<String, IpAddr> = dns_overrides
        .iter()
        .map(|(host, address)| (host.to_ascii_lowercase(), *address))
        .collect();
    let builder = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .no_proxy();
    let Some(filter) = filter else {
        let mut builder = builder.redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS));
        for (host, address) in &overrides {
            builder = builder.resolve(host, SocketAddr::new(*address, 0));
        }
        let client = builder.build().map_err(|error| {
            ApiError::internal(format!("cannot build the image client: {error}"))
        })?;
        return Ok(ImageFetcherImpl::Plain(ReqwestImageFetcher::from_client(
            client,
            max_image_bytes,
        )));
    };
    let resolver = FilteringOverrideResolver {
        filter: Arc::clone(&filter),
        overrides,
    };
    let client = builder
        .dns_resolver(Arc::new(resolver))
        .redirect(filtered_redirect_policy(Arc::clone(&filter)))
        .build()
        .map_err(|error| ApiError::internal(format!("cannot build the image client: {error}")))?;
    let inner = ReqwestImageFetcher::from_client(client, max_image_bytes);
    Ok(ImageFetcherImpl::Filtered(FilteredFetcher {
        inner,
        filter,
    }))
}

/// A fetcher that applies the destination filter to the initial URL, every
/// resolved address and every redirect.
struct FilteredFetcher {
    inner: ReqwestImageFetcher,
    filter: Arc<dyn DestinationFilter>,
}

impl ImageFetcher for FilteredFetcher {
    async fn fetch(&self, url: &str, budget: &ImageByteBudget) -> Result<Vec<u8>, ImageError> {
        // A host that is already an IP address skips the DNS resolver, so check it
        // here as well as in the redirect policy.
        if let Ok(parsed) = reqwest::Url::parse(url)
            && let Err(rejected) = validate_ip_address_host(&parsed, self.filter.as_ref())
        {
            return Err(ImageError::destination_rejected(url, rejected.host));
        }
        match self.inner.fetch(url, budget).await {
            Err(error) => {
                if let ImageError::Fetch { source, .. } = &error
                    && let Some(host) = rejected_host(source.as_ref())
                {
                    return Err(ImageError::destination_rejected(url, host));
                }
                Err(error)
            }
            ok => ok,
        }
    }
}

/// The resolver of the filtered client: it applies the test-only name overrides
/// (empty in production) before the destination filter, so an overridden name
/// cannot bypass the filter, and returns only the addresses the filter allows.
struct FilteringOverrideResolver {
    filter: Arc<dyn DestinationFilter>,
    overrides: HashMap<String, IpAddr>,
}

impl Resolve for FilteringOverrideResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_ascii_lowercase();
        let override_address = self.overrides.get(&host).copied();
        let filter = Arc::clone(&self.filter);
        Box::pin(async move {
            let addresses: Vec<SocketAddr> = match override_address {
                Some(address) => vec![SocketAddr::new(address, 0)],
                None => tokio::net::lookup_host((host.as_str(), 0)).await?.collect(),
            };
            let allowed: Vec<SocketAddr> = addresses
                .into_iter()
                .filter(|address| filter.allows(&host, address.ip()))
                .collect();
            if allowed.is_empty() {
                return Err(Box::new(DestinationRejected { host }) as _);
            }
            Ok(Box::new(allowed.into_iter()) as _)
        })
    }
}

/// A redirect policy that checks IP-address redirect targets.  Host names are
/// checked by the resolver when the client resolves them.
fn filtered_redirect_policy(filter: Arc<dyn DestinationFilter>) -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(move |attempt| {
        // The first entry of `previous` is the initial URL, as in `Policy::limited`.
        if attempt.previous().len() > MAX_REDIRECTS {
            return attempt.error(TooManyRedirects);
        }
        match validate_ip_address_host(attempt.url(), filter.as_ref()) {
            Ok(()) => attempt.follow(),
            Err(rejected) => attempt.error(rejected),
        }
    })
}

#[derive(Debug)]
struct DestinationRejected {
    host: String,
}

impl fmt::Display for DestinationRejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "the destination filter rejects {}", self.host)
    }
}

impl StdError for DestinationRejected {}

#[derive(Debug)]
struct TooManyRedirects;

impl fmt::Display for TooManyRedirects {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "more than 5 redirects")
    }
}

impl StdError for TooManyRedirects {}

fn validate_ip_address_host(
    url: &reqwest::Url,
    filter: &dyn DestinationFilter,
) -> Result<(), DestinationRejected> {
    let Some(host) = url.host_str() else {
        return Ok(());
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    match host.parse::<IpAddr>() {
        Ok(address) if !filter.allows(host, address) => Err(DestinationRejected {
            host: host.to_owned(),
        }),
        _ => Ok(()),
    }
}

fn rejected_host(error: &(dyn StdError + 'static)) -> Option<String> {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(rejected) = error.downcast_ref::<DestinationRejected>() {
            return Some(rejected.host.clone());
        }
        current = error.source();
    }
    None
}

/// Whether the destination filter allows `address` for `host`.
pub fn destination_allowed(_host: &str, address: IpAddr) -> bool {
    match address.to_canonical() {
        IpAddr::V4(address) => allowed_v4(address),
        IpAddr::V6(address) => allowed_v6(address),
    }
}

fn allowed_v4(address: Ipv4Addr) -> bool {
    let octets = address.octets();
    !(address.is_unspecified()
        || octets[0] == 0
        || address.is_loopback()
        || address.is_private()
        || address.is_link_local()
        || address.is_multicast()
        || address.is_broadcast()
        || octets[0] >= 240
        || (octets[0] == 100 && octets[1] & 0xc0 == 64)
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
        || (octets[0] == 192 && octets[1] == 31 && octets[2] == 196)
        || (octets[0] == 192 && octets[1] == 52 && octets[2] == 193)
        || (octets[0] == 192 && octets[1] == 88 && octets[2] == 99)
        || (octets[0] == 192 && octets[1] == 175 && octets[2] == 48)
        || (octets[0] == 198 && octets[1] == 18)
        || (octets[0] == 198 && octets[1] == 19)
        || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
        || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113))
}

fn allowed_v6(address: Ipv6Addr) -> bool {
    let segments = address.segments();
    if address.is_unspecified()
        || address.is_loopback()
        || address.is_unique_local()
        || address.is_unicast_link_local()
        || address.is_multicast()
    {
        return false;
    }
    if let Some(address) = address.to_ipv4_mapped() {
        return allowed_v4(address);
    }
    if segments[0] == 0x0064 && segments[1] == 0xff9b {
        // 64:ff9b::/96 embeds an IPv4 address; 64:ff9b:1::/48 is local-use NAT64.
        if segments[2] == 0 && segments[3] == 0 && segments[4] == 0 && segments[5] == 0 {
            let embedded = Ipv4Addr::new(
                (segments[6] >> 8) as u8,
                segments[6] as u8,
                (segments[7] >> 8) as u8,
                segments[7] as u8,
            );
            return allowed_v4(embedded);
        }
        if segments[2] == 1 {
            return false;
        }
    }
    // 2001::/23: IETF protocol assignments, including Teredo and ORCHID.
    if segments[0] == 0x2001 && segments[1] & 0xfe00 == 0 {
        return false;
    }
    // 2001:db8::/32 documentation.
    if segments[0] == 0x2001 && segments[1] == 0x0db8 {
        return false;
    }
    // 2002::/16 6to4 embeds an IPv4 address.
    if segments[0] == 0x2002 {
        let embedded = Ipv4Addr::new(
            (segments[1] >> 8) as u8,
            segments[1] as u8,
            (segments[2] >> 8) as u8,
            segments[2] as u8,
        );
        return allowed_v4(embedded);
    }
    // 100::/64 discard-only.
    if segments[0] == 0x0100 && segments[1] == 0 && segments[2] == 0 && segments[3] == 0 {
        return false;
    }
    // fec0::/10 deprecated site-local.
    if segments[0] & 0xffc0 == 0xfec0 {
        return false;
    }
    // 3fff::/20 documentation.
    if segments[0] == 0x3fff && segments[1] & 0xf000 == 0 {
        return false;
    }
    true
}

/// A preprocessor that reads the header dimensions and keeps the original bytes.
struct HeaderPreprocessor;

impl ImagePreprocessor for HeaderPreprocessor {
    async fn preprocess(
        &self,
        data: Vec<u8>,
        options: PreprocessOptions,
    ) -> Result<ImageInfo, ImageError> {
        if data.is_empty() {
            return Err(ImageError::empty_image());
        }
        let format = image::guess_format(&data)
            .map_err(|_| ImageError::unsupported_media_type("unknown media type"))?;
        match format {
            image::ImageFormat::Png
            | image::ImageFormat::Jpeg
            | image::ImageFormat::Gif
            | image::ImageFormat::WebP => {}
            other => return Err(ImageError::unsupported_media_type(other.to_mime_type())),
        }
        let reader = image::ImageReader::with_format(Cursor::new(&data), format);
        let (width, height) = reader
            .into_dimensions()
            .map_err(|error| ImageError::decode(error.to_string()))?;
        if width == 0 || height == 0 {
            return Err(ImageError::empty_image());
        }
        if width > options.max_dimension_px || height > options.max_dimension_px {
            return Err(ImageError::dimensions_too_large());
        }
        let spec = ImageTokenSpec::v41();
        if let Err(error) = spec.calc_token_count(width as usize, height as usize) {
            return Err(ImageError::TokenBudget(error));
        }
        Ok(ImageInfo {
            data,
            width,
            height,
        })
    }
}

fn image_error(error: ImageError) -> ApiError {
    match error {
        ImageError::DestinationRejected { host, .. } => {
            ApiError::bad_request(format!("the destination filter rejects {host}"))
        }
        ImageError::Client(_) => ApiError::internal(error.to_string()),
        _ => ApiError::bad_request(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use deepseek_recipe_core::multimodal::{CalcResizeError, ResizeResult};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    /// A loopback server for the fetcher tests: `/image.png` is a small PNG,
    /// `/to/<n>` redirects to the n-th entry of `locations`, and `/chain/<n>`
    /// redirects to `/chain/<n-1>` until `/chain/0`, which is the PNG.  It counts
    /// the requests that reach it, so a rejection can be shown to happen before
    /// any connection.
    async fn spawn_server(
        locations: impl FnOnce(u16) -> Vec<String>,
    ) -> (u16, Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind");
        let port = listener.local_addr().expect("address").port();
        let locations = locations(port);
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&hits);
        let png = {
            let image = image::RgbaImage::from_pixel(16, 16, image::Rgba([255, 0, 0, 255]));
            let mut bytes = Vec::new();
            image::DynamicImage::ImageRgba8(image)
                .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
                .expect("encode PNG");
            bytes
        };
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                counter.fetch_add(1, Ordering::SeqCst);
                let locations = locations.clone();
                let png = png.clone();
                tokio::spawn(async move {
                    let mut request = [0; 2048];
                    let read = stream.read(&mut request).await.unwrap_or(0);
                    let request = String::from_utf8_lossy(&request[..read]);
                    let path = request.split_whitespace().nth(1).unwrap_or("/").to_owned();
                    let redirect = |location: String| {
                        format!(
                            "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                        .into_bytes()
                    };
                    let response = if let Some(n) = path.strip_prefix("/to/") {
                        redirect(locations[n.parse::<usize>().unwrap()].clone())
                    } else if let Some(n) = path.strip_prefix("/chain/")
                        && let Ok(n) = n.parse::<usize>()
                        && n > 0
                    {
                        redirect(format!("/chain/{}", n - 1))
                    } else {
                        let mut response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            png.len()
                        )
                        .into_bytes();
                        response.extend_from_slice(&png);
                        response
                    };
                    let _ = stream.write_all(&response).await;
                });
            }
        });
        (port, hits)
    }

    /// The production fetcher with a filter that also admits `allowed.test`, the
    /// one name that may reach the loopback test server.  Every other name and
    /// address goes through `destination_allowed`, as in production.
    fn test_fetcher() -> ImageFetcherImpl {
        let filter: Arc<dyn DestinationFilter> = Arc::new(|host: &str, address: IpAddr| {
            host == "allowed.test" || destination_allowed(host, address)
        });
        let overrides = [
            ("allowed.test", IpAddr::V4(Ipv4Addr::LOCALHOST)),
            ("blocked.test", IpAddr::V4(Ipv4Addr::LOCALHOST)),
            ("mapped.test", "::ffff:127.0.0.1".parse().unwrap()),
        ]
        .into_iter()
        .map(|(host, address)| (host.to_owned(), address))
        .collect();
        build_fetcher_with(Some(filter), &overrides, 1 << 20).expect("fetcher")
    }

    fn budget() -> ImageByteBudget {
        ImageByteBudget::new(1 << 20, 1 << 20, 0)
    }

    fn rejected(result: Result<Vec<u8>, ImageError>) -> String {
        match result {
            Err(ImageError::DestinationRejected { host, .. }) => host,
            other => panic!("expected a destination rejection, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn the_production_fetcher_reaches_an_allowed_destination() {
        // Positive control: without it, the rejections below could be any failure.
        let (port, _) = spawn_server(|_| Vec::new()).await;
        let fetcher = test_fetcher();
        let bytes = fetcher
            .fetch(&format!("http://allowed.test:{port}/image.png"), &budget())
            .await
            .expect("allowed fetch");
        assert!(bytes.starts_with(b"\x89PNG"));
        let bytes = fetcher
            .fetch(
                &format!("http://allowed.test:{port}/chain/{MAX_REDIRECTS}"),
                &budget(),
            )
            .await
            .expect("five redirects are followed");
        assert!(bytes.starts_with(b"\x89PNG"));
    }

    #[tokio::test]
    async fn a_sixth_redirect_is_refused() {
        let (port, _) = spawn_server(|_| Vec::new()).await;
        let error = test_fetcher()
            .fetch(
                &format!("http://allowed.test:{port}/chain/{}", MAX_REDIRECTS + 1),
                &budget(),
            )
            .await
            .expect_err("too many redirects");
        assert!(
            matches!(error, ImageError::Fetch { .. }),
            "expected a fetch error, got {error:?}"
        );
    }

    #[tokio::test]
    async fn every_redirect_target_is_checked_before_connecting() {
        // Each redirect leaves an allowed host for a loopback destination spelled a
        // different way. The server must see only the first request of each case.
        let targets = |port: u16| {
            vec![
                (format!("http://127.0.0.1:{port}/image.png"), "127.0.0.1"),
                (format!("http://2130706433:{port}/image.png"), "127.0.0.1"),
                (format!("http://0x7f.1:{port}/image.png"), "127.0.0.1"),
                (
                    format!("http://[::ffff:127.0.0.1]:{port}/image.png"),
                    "::ffff:7f00:1",
                ),
                (format!("http://[::1]:{port}/image.png"), "::1"),
                (
                    format!("http://blocked.test:{port}/image.png"),
                    "blocked.test",
                ),
                (
                    format!("http://mapped.test:{port}/image.png"),
                    "mapped.test",
                ),
                (format!("http://localhost:{port}/image.png"), "localhost"),
            ]
        };
        let (port, hits) =
            spawn_server(|port| targets(port).into_iter().map(|(url, _)| url).collect()).await;
        let targets = targets(port);
        let fetcher = test_fetcher();
        for (index, (url, host)) in targets.iter().enumerate() {
            let before = hits.load(std::sync::atomic::Ordering::SeqCst);
            let result = fetcher
                .fetch(&format!("http://allowed.test:{port}/to/{index}"), &budget())
                .await;
            assert_eq!(rejected(result), *host, "redirect to {url}");
            let after = hits.load(std::sync::atomic::Ordering::SeqCst);
            assert_eq!(after - before, 1, "redirect to {url} reached the server");
        }
    }

    #[tokio::test]
    async fn an_initial_url_spelled_as_an_address_is_rejected_without_connecting() {
        let (port, hits) = spawn_server(|_| Vec::new()).await;
        let fetcher = test_fetcher();
        for (url, host) in [
            (format!("http://127.0.0.1:{port}/image.png"), "127.0.0.1"),
            (format!("http://2130706433:{port}/image.png"), "127.0.0.1"),
            (
                format!("http://[::ffff:127.0.0.1]:{port}/image.png"),
                "::ffff:7f00:1",
            ),
            (
                format!("http://blocked.test:{port}/image.png"),
                "blocked.test",
            ),
        ] {
            assert_eq!(
                rejected(fetcher.fetch(&url, &budget()).await),
                host,
                "{url}"
            );
        }
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn a_token_budget_error_is_the_api_400() {
        let error = ImageError::TokenBudget(CalcResizeError::NotConverged {
            max_iter: 10,
            last_result: ResizeResult {
                n_llm_h: 1,
                n_llm_w: 1,
                best_height: 1,
                best_width: 1,
                token_count: 4,
            },
        });
        let api = image_error(error);
        assert_eq!(api.status, 400);
        assert_eq!(api.body["error"]["type"], "invalid_request_error");
        assert!(
            api.body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("did not converge")
        );
    }
}
