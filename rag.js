import { pipeline, cos_sim } from '@xenova/transformers';

class RAGSystem {
    constructor() {
        this.embedder = null;
        this.documents = [];
        this.embeddings = [];
    }

    async initialize() {
        // Initialize the embedding model
        this.embedder = await pipeline('feature-extraction', 'Xenova/jina-embeddings-v2-base-en', {
            quantized: false
        });
    }

    // Function to extract text from PDF
    async extractTextFromPDF(pdfFile) {
        try {
            const arrayBuffer = await pdfFile.arrayBuffer();
            const loadingTask = pdfjsLib.getDocument({
                data: arrayBuffer,
                verbosity: 0
            });

            const pdf = await loadingTask.promise;
            let fullText = '';
            
            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const textContent = await page.getTextContent();
                const pageText = textContent.items.map(item => item.str).join(' ');
                fullText += pageText + ' ';
            }

            return fullText;
        } catch (error) {
            console.error('Error extracting PDF text:', error);
            throw new Error('Failed to extract text from PDF');
        }
    }

    // Function to split text into chunks
    splitIntoChunks(text, maxChunkSize = 500) {
        // First split by sentences
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        const chunks = [];
        let currentChunk = '';

        for (const sentence of sentences) {
            if ((currentChunk + sentence).length <= maxChunkSize) {
                currentChunk += sentence + ' ';
            } else {
                if (currentChunk) {
                    chunks.push(currentChunk.trim());
                }
                currentChunk = sentence + ' ';
            }
        }

        if (currentChunk) {
            chunks.push(currentChunk.trim());
        }

        return chunks;
    }

    // Function to process new text content
    async processText(text) {
        try {
            // Split text into chunks
            const chunks = this.splitIntoChunks(text);
            
            // Store the chunks
            this.documents.push(...chunks);
            
            // Compute embeddings for each chunk
            for (const chunk of chunks) {
                const output = await this.embedder(chunk, { pooling: 'mean' });
                this.embeddings.push({
                    text: chunk,
                    embedding: output.data
                });
            }
            
            return chunks.length;
        } catch (error) {
            console.error('Error processing text:', error);
            throw error;
        }
    }

    // Function to process PDF file
    async processPDF(file) {
        try {
            // Extract text from PDF
            const text = await this.extractTextFromPDF(file);
            
            // Process the extracted text
            return await this.processText(text);
        } catch (error) {
            console.error('Error processing PDF:', error);
            throw error;
        }
    }

    // Function to find relevant context for a query
    async findRelevantContext(query, topK = 3) {
        if (this.embeddings.length === 0) {
            return [];
        }

        // Get query embedding
        const queryOutput = await this.embedder(query, { pooling: 'mean' });
        
        // Calculate similarities
        const similarities = this.embeddings.map((doc, index) => ({
            index,
            score: cos_sim(queryOutput.data, doc.embedding),
            text: doc.text
        }));

        // Sort by similarity score
        similarities.sort((a, b) => b.score - a.score);
        
        // Return top K most similar chunks
        return similarities.slice(0, topK).map(item => item.text);
    }

    // Clear all stored documents and embeddings
    clear() {
        this.documents = [];
        this.embeddings = [];
    }
}

export const ragSystem = new RAGSystem();
