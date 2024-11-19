import { Phi3SLM } from './phi3_slm.js';
import { env, AutoTokenizer,pipeline, cos_sim } from '@xenova/transformers';

export class RAG {
    extractor = undefined;
    phi3_slm = undefined;
    tokenizer = undefined;

    constructor(){
        this.phi3_slm = new Phi3SLM();
    }

    async InitPhi3SLM(){
        try {
            this.tokenizer = await AutoTokenizer.from_pretrained("Microsoft/Phi-3-mini-4k-instruct-onnx-web");
            await this.phi3_slm.loadONNX();
            await this.load('Xenova/jina-embeddings-v2-base-en');
        } catch (error) {
            console.error('Error initializing Phi3SLM:', error);
            throw error;
        }
    }

    async loadONNX(){
        try {
            await this.phi3_slm.loadONNX();
        } catch (error) {
            console.error('Error loading ONNX:', error);
            throw error;
        }
    }

    async load(embeddd_model) {
        try {
            this.extractor = await pipeline('feature-extraction', embeddd_model, { 
                quantized: false 
            });
        } catch (error) {
            console.error('Error loading embedding model:', error);
            throw error;
        }
    }

    async getEmbeddings(query, kbContents) { 
        try {
            if (!query || !kbContents || kbContents.length === 0) {
                throw new Error('Invalid input for embeddings');
            }

            const question = query;
            let sim_result = [];

            for(const content of kbContents) {
                const output = await this.extractor([question, content], { pooling: 'mean' });
                const sim = cos_sim(output[0].data, output[1].data);
                sim_result.push({ content, sim });
            }

            sim_result.sort((a, b) => b.sim - a.sim);

            if (sim_result.length === 0) {
                throw new Error('No matching content found');
            }

            return sim_result[0].content;
        } catch (error) {
            console.error('Error in getEmbeddings:', error);
            throw error;
        }
    }

    async generateSummaryContent(prompt) {
        try {
            if (!prompt) {
                throw new Error('Invalid prompt');
            }

            const { input_ids } = await this.tokenizer(prompt, { 
                return_tensor: false, 
                padding: true, 
                truncation: true 
            });

            // Clear previous state
            this.phi3_slm.initilize_feed();
            
            const start_timer = performance.now();
            const output_index = this.phi3_slm.output_tokens.length + input_ids.length;
            let answer_result = '';
            
            const output_tokens = await this.phi3_slm.generate(input_ids, (output_tokens) => {
                if (output_tokens.length == input_ids.length + 1) {
                    const took = (performance.now() - start_timer) / 1000;
                    console.log(`time to first token in ${took.toFixed(1)}sec, ${input_ids.length} tokens`);
                }
                answer_result = this.token_to_text(this.tokenizer, output_tokens, output_index);
            }, { 
                max_tokens: 4000,
                temperature: 0.7, // Add temperature for more stable outputs
                top_p: 0.9 // Add top_p for better sampling
            });

            const took = (performance.now() - start_timer) / 1000;
            const seqlen = output_tokens.length - output_index;
            console.log(`${seqlen} tokens in ${took.toFixed(1)}sec, ${(seqlen / took).toFixed(2)} tokens/sec`);

            // Force cleanup
            this.phi3_slm.initilize_feed();
            
            if (!answer_result) {
                throw new Error('Failed to generate summary');
            }

            return answer_result;
        } catch (error) {
            console.error('Error in generateSummaryContent:', error);
            throw error;
        }
    }

    async generateEmbeddingsContent(prompt) {
        try {
            if (!prompt) {
                throw new Error('Invalid prompt');
            }

            const { input_ids } = await this.tokenizer(prompt, { 
                return_tensor: false, 
                padding: true, 
                truncation: true 
            });

            // Clear previous state
            this.phi3_slm.initilize_feed();

            const start_timer = performance.now();
            const output_index = this.phi3_slm.output_tokens.length + input_ids.length;
            let json_result = '';
            
            const output_tokens = await this.phi3_slm.generate(input_ids, (output_tokens) => {
                if (output_tokens.length == input_ids.length + 1) {
                    const took = (performance.now() - start_timer) / 1000;
                    console.log(`time to first token in ${took.toFixed(1)}sec, ${input_ids.length} tokens`);
                }
                json_result = this.token_to_text(this.tokenizer, output_tokens, output_index);
            }, { 
                max_tokens: 4000,
                temperature: 0.7,
                top_p: 0.9
            });

            const took = (performance.now() - start_timer) / 1000;
            const seqlen = output_tokens.length - output_index;
            console.log(`${seqlen} tokens in ${took.toFixed(1)}sec, ${(seqlen / took).toFixed(2)} tokens/sec`);

            const json_result_index = json_result.indexOf(']');
            if (json_result_index === -1) {
                throw new Error('Invalid JSON result format');
            }

            json_result = json_result.substring(0, json_result_index + 1);

            // Force cleanup
            this.phi3_slm.initilize_feed();

            return json_result;
        } catch (error) {
            console.error('Error in generateEmbeddingsContent:', error);
            throw error;
        }
    }

    token_to_text(tokenizer, tokens, startidx) {
        try {
            return this.tokenizer.decode(tokens.slice(startidx), { 
                skip_special_tokens: true 
            });
        } catch (error) {
            console.error('Error in token_to_text:', error);
            throw error;
        }
    }
}
