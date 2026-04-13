use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::transport::Server;

mod engine;
mod service;
mod vad;

pub mod proto {
    tonic::include_proto!("alignment");
}

use engine::{AlignmentEngine, EngineConfig};
use proto::alignment_engine_server::AlignmentEngineServer;
use service::AlignmentService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::]:50051".parse()?;

    let engine = Arc::new(Mutex::new(AlignmentEngine::new(EngineConfig::default())));
    let svc = AlignmentService::new(engine);

    eprintln!("alass-service listening on {}", addr);

    Server::builder()
        .add_service(AlignmentEngineServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
