use std::env;
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
    // Fix #8: Accept address from --addr flag or ALASS_ADDR env var
    let addr_str = parse_addr_arg()
        .or_else(|| env::var("ALASS_ADDR").ok())
        .unwrap_or_else(|| "[::]:50051".to_string());
    let addr = addr_str.parse()?;

    let engine = Arc::new(Mutex::new(AlignmentEngine::new(EngineConfig::default())));
    let svc = AlignmentService::new(engine);

    eprintln!("alass-service listening on {}", addr);

    // Fix #10: Graceful shutdown on SIGINT/SIGTERM
    Server::builder()
        .add_service(AlignmentEngineServer::new(svc))
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

    eprintln!("alass-service shut down gracefully");
    Ok(())
}

fn parse_addr_arg() -> Option<String> {
    let args: Vec<String> = env::args().collect();
    for (i, arg) in args.iter().enumerate() {
        if arg == "--addr" {
            return args.get(i + 1).cloned();
        }
        if let Some(val) = arg.strip_prefix("--addr=") {
            return Some(val.to_string());
        }
    }
    None
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    {
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => { eprintln!("\nreceived SIGINT, shutting down..."); }
            _ = sigterm.recv() => { eprintln!("received SIGTERM, shutting down..."); }
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await.expect("failed to listen for ctrl-c");
        eprintln!("\nreceived SIGINT, shutting down...");
    }
}
