use anyhow::Result;
use serde_json::{json, Value};
use std::collections::HashMap;
use tokio::time::{timeout, Duration};
use reqwest::Client;
use crate::database::{Database, DocumentEmbedding, DocumentPage};

#[derive(Clone)]
pub struct EmbeddingService {
    client: Client,
    api_key: Option<String>,
    model: String,
    cache: HashMap<String, Vec<f32>>,
}

impl EmbeddingService {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            model: "text-embedding-3-small".to_string(),
            cache: HashMap::new(),
        }
    }

    /// Generate embeddings for text content
    pub async fn generate_embedding(&mut self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }

        // If no API key, return a simple hash-based embedding for demo
        if self.api_key.is_none() {
            return Ok(self.generate_simple_embedding(text));
        }

        // Call OpenAI API for real embeddings
        let embedding = self.call_openai_embedding(text).await?;
        
        // Cache the result
        self.cache.insert(text.to_string(), embedding.clone());
        
        Ok(embedding)
    }

    /// Generate embeddings for document chunks
    pub async fn generate_document_embeddings(&mut self, db: &Database, page: &DocumentPage) -> Result<Vec<DocumentEmbedding>> {
        let chunks = self.chunk_document(&page.content, 512);
        let mut embeddings = Vec::new();

        for (index, chunk) in chunks.iter().enumerate() {
            let embedding_vec = self.generate_embedding(chunk).await?;
            
            let embedding = DocumentEmbedding {
                id: None,
                page_id: page.id.clone(),
                embedding_model: self.model.clone(),
                embedding: embedding_vec,
                chunk_index: index as i32,
                chunk_text: chunk.clone(),
                created_at: chrono::Utc::now(),
            };
            
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Perform semantic search using vector similarity
    pub async fn semantic_search(&mut self, db: &Database, query: &str, limit: usize) -> Result<Vec<(String, f32)>> {
        let query_embedding = self.generate_embedding(query).await?;
        let similar_chunks = db.find_similar_embeddings(&query_embedding, limit * 3).await?;
        
        // Group by page and calculate average similarity
        let mut page_scores: HashMap<String, Vec<f32>> = HashMap::new();
        
        for (page_id, similarity) in similar_chunks {
            page_scores.entry(page_id).or_default().push(similarity);
        }
        
        // Calculate average scores and sort
        let mut results: Vec<(String, f32)> = page_scores
            .into_iter()
            .map(|(page_id, scores)| {
                let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
                (page_id, avg_score)
            })
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(limit);
        
        Ok(results)
    }

    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Chunk document content for embedding generation
    fn chunk_document(&self, content: &str, max_tokens: usize) -> Vec<String> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut current_length = 0;

        for word in words {
            let word_length = word.len() + 1; // +1 for space
            
            if current_length + word_length > max_tokens && !current_chunk.is_empty() {
                chunks.push(current_chunk.join(" "));
                current_chunk.clear();
                current_length = 0;
            }
            
            current_chunk.push(word);
            current_length += word_length;
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk.join(" "));
        }

        chunks
    }

    /// Generate simple hash-based embedding for demo purposes
    fn generate_simple_embedding(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Convert hash to 384-dimensional vector (same as text-embedding-3-small)
        let mut embedding = vec![0.0; 384];
        for i in 0..384 {
            let byte_index = i % 8;
            let byte_val = ((hash >> (byte_index * 8)) & 0xFF) as f32;
            embedding[i] = (byte_val - 127.5) / 127.5; // Normalize to [-1, 1]
        }
        
        // Add some text-based features
        let word_count = text.split_whitespace().count() as f32;
        let char_count = text.len() as f32;
        let avg_word_length = if word_count > 0.0 { char_count / word_count } else { 0.0 };
        
        // Modify some dimensions based on text features
        if embedding.len() > 3 {
            embedding[0] = (word_count / 100.0).tanh(); // Word count feature
            embedding[1] = (char_count / 1000.0).tanh(); // Character count feature
            embedding[2] = (avg_word_length / 10.0).tanh(); // Average word length
        }
        
        embedding
    }

    /// Call OpenAI API for embeddings
    async fn call_openai_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let api_key = self.api_key.as_ref()
            .ok_or_else(|| anyhow::anyhow!("OpenAI API key not found"))?;

        let request_body = json!({
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        });

        let response = timeout(Duration::from_secs(30), 
            self.client
                .post("https://api.openai.com/v1/embeddings")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send()
        ).await??;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;
        
        let embedding = response_json["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid embedding response format"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(embedding)
    }

    /// Batch process embeddings for multiple documents
    pub async fn batch_generate_embeddings(&mut self, db: &Database, pages: Vec<DocumentPage>) -> Result<()> {
        for page in pages {
            let embeddings = self.generate_document_embeddings(db, &page).await?;
            
            for embedding in embeddings {
                db.store_embedding(&embedding).await?;
            }
            
            // Small delay to respect rate limits
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_embedding_generation() {
        let mut service = EmbeddingService::new();
        let embedding = service.generate_simple_embedding("Hello world test");
        
        assert_eq!(embedding.len(), 384);
        assert!(embedding.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_cosine_similarity() {
        let service = EmbeddingService::new();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        
        assert!((service.cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!((service.cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_document_chunking() {
        let service = EmbeddingService::new();
        let content = "This is a test document with many words that should be chunked properly.";
        let chunks = service.chunk_document(content, 20);
        
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|chunk| chunk.len() <= 25)); // Some tolerance for word boundaries
    }
}
