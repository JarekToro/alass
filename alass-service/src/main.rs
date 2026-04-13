mod engine;
mod grpc;
mod proto;
mod subtitle;
mod warp;

use crate::grpc::AlignmentService;
use crate::proto::alignment::alignment_engine_server::AlignmentEngineServer;
use std::net::SocketAddr;
use tonic::transport::Server;

const DEFAULT_ADDR: &str = "0.0.0.0:50051";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let addr: SocketAddr = std::env::var("ALASS_SERVICE_ADDR")
        .unwrap_or_else(|_| DEFAULT_ADDR.to_string())
        .parse()
        .expect("invalid bind address");

    let service = AlignmentService::new();

    log::info!("alass-service listening on {}", addr);

    Server::builder()
        .add_service(AlignmentEngineServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
