-- PostgreSQL initialization script for RAG system
-- Creates pgvector extension and required tables

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT,
    file_size INTEGER,
    page_count INTEGER,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Chunks table with vector embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding VECTOR(768),  -- nomic-embed-text produces 768-dim vectors
    chunk_index INTEGER,
    page_number INTEGER,
    section_title TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Query history table
CREATE TABLE IF NOT EXISTS query_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_text TEXT,
    sources JSONB,
    chain_of_thought TEXT,
    feedback TEXT CHECK (feedback IN ('helpful', 'neutral', 'not_helpful')),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for efficient search
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_query_history_created ON query_history(created_at DESC);

-- Function to update chunk count on documents
CREATE OR REPLACE FUNCTION update_chunk_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE documents SET chunk_count = chunk_count + 1, updated_at = NOW()
        WHERE id = NEW.document_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE documents SET chunk_count = chunk_count - 1, updated_at = NOW()
        WHERE id = OLD.document_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger for chunk count
DROP TRIGGER IF EXISTS trigger_update_chunk_count ON chunks;
CREATE TRIGGER trigger_update_chunk_count
AFTER INSERT OR DELETE ON chunks
FOR EACH ROW EXECUTE FUNCTION update_chunk_count();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag;
