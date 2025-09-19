use std::fs;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use hnsw_rs::prelude::*;
use std::path::Path;

fn main() {
    // Step 1: Load embedding model
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
        .create_model().unwrap();

    let sentences_file_path ="files/sentences_2.txt";
    let contents = fs::read_to_string(sentences_file_path).unwrap();

    let sentences: Vec<&str> = contents.lines().collect();

    let embeddings = model.encode(&sentences).unwrap();

    // Step 3: Build HNSW index
    let nb_conn = 16;
    let max_item = 10_000;
    let ef_c = 200;
    let dim = embeddings[0].len();

    let hnsw = Hnsw::<f32, DistCosine>::new(nb_conn, max_item, ef_c, dim, DistCosine {});

    // Insert vectors into index
    for (i, emb) in embeddings.iter().enumerate() {
        println!("Sentence: {} \nEmbedding: \n{:?}", sentences[i], emb);
        hnsw.insert((emb, i));
    }

    // Step 4: Persist index
    hnsw.file_dump(Path::new("./db/"), "sentences").unwrap();

    // Step 5: Run a semantic search
    let query = "I love japanese food";
    let query_emb = model.encode(&[query]).unwrap();

    println!("Query Embedding:\n{:?}", query_emb);


    let ef_search = 200;
    let num_results = 5;
    let results = hnsw.search(&query_emb[0], num_results, ef_search);

    println!("\nQuery: \"{}\"", query);
    for r in results {
        let idx = r.d_id as usize;
        println!("→ \"{}\" (score: {:.6})", sentences[idx], r.distance);
    }

}
