import { env, AutoTokenizer,pipeline, cos_sim } from '@xenova/transformers';
import { RAG } from './rag.js';
import { Phi3SLM } from './phi3_slm.js';
import { marked } from 'marked';

const fileContents = [];
const kbContents = [];
let rag;
let embeddingWorker;
let isProcessing = false;

document.getElementById('send-button').addEventListener('click', async function() {
    if (isProcessing) {
        console.log('Already processing a request, please wait...');
        return;
    }

    const inputBox = document.querySelector('.input-box');
    const question = inputBox.value.trim();
    
    if (question) {
        try {
            isProcessing = true;
            const chatContainer = document.querySelector('.chat-container');
            
            // Add user message
            const userMessage = document.createElement('div');
            userMessage.className = 'message question';
            userMessage.textContent = question;
            chatContainer.appendChild(userMessage);

            // Clear input
            inputBox.value = '';

            // Show loading indicator
            const loadingMessage = document.createElement('div');
            loadingMessage.className = 'message answer loading';
            loadingMessage.textContent = 'Processing...';
            chatContainer.appendChild(loadingMessage);

            try {
                const answer = await generateAnswer(question);
                const summary = await generativeSummary(answer);

                // Remove loading message
                chatContainer.removeChild(loadingMessage);

                // Add bot response
                const botMessage = document.createElement('div');
                botMessage.className = 'message answer';
                botMessage.textContent = summary;
                chatContainer.appendChild(botMessage);
            } catch (error) {
                // Remove loading message
                chatContainer.removeChild(loadingMessage);

                // Show error message
                const errorMessage = document.createElement('div');
                errorMessage.className = 'message answer error';
                errorMessage.textContent = 'Sorry, there was an error processing your request. Please try again.';
                chatContainer.appendChild(errorMessage);
                
                console.error('Error processing request:', error);
            }

            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
        } finally {
            isProcessing = false;
        }
    }
});

async function generateAnswer(question) {
    try {
        if (!rag || !question) {
            throw new Error('RAG not initialized or invalid question');
        }
        return await rag.getEmbeddings(question, kbContents);
    } catch (error) {
        console.error('Error in generateAnswer:', error);
        throw error;
    }
}

async function generativeSummary(answer) {
    try {
        if (!rag || !answer) {
            throw new Error('RAG not initialized or invalid answer');
        }
        const prompt = `<|system|>\nYou are a friendly assistant. Help me summarize answer of the knowledge points<|end|>\n<|user|>\n${answer}<|end|>\n<|assistant|>\n`;
        return await rag.generateSummaryContent(prompt);
    } catch (error) {
        console.error('Error in generativeSummary:', error);
        throw error;
    }
}

async function initializeWorker() {
    embeddingWorker = new Worker(new URL('./embedding.worker.js', import.meta.url), { type: 'module' });
    
    embeddingWorker.onmessage = function(e) {
        const data = e.data;
        
        switch(data.type) {
            case 'initialized':
                console.log('Worker initialized');
                break;
            case 'progress':
                updateProgress(data.progress);
                break;
            case 'complete':
                kbContents.push(...data.result);
                if (kbContents.length > 0) {
                    document.getElementById('chat-message').style.display = 'block';
                    document.getElementById('progress-container').style.display = 'none';
                }
                break;
            case 'error':
                console.error('Worker error:', data.error);
                document.getElementById('progress-container').style.display = 'none';
                alert('Error processing file: ' + data.error);
                break;
        }
    };

    embeddingWorker.onerror = function(error) {
        console.error('Worker error:', error);
        document.getElementById('progress-container').style.display = 'none';
        alert('Error initializing worker: ' + error.message);
    };

    embeddingWorker.postMessage({ type: 'init' });
}

function updateProgress(progress) {
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
    }
}

document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('file-input');
    const files = fileInput.files;

    if (files.length === 0) {
        alert('Please upload at least one file.');
        return;
    }

    try {
        document.getElementById('progress-container').style.display = 'block';

        for (const file of files) {
            const text = await file.text();
            fileContents.push(text);
        }

        embeddingWorker.postMessage({ 
            type: 'process', 
            content: fileContents[0] 
        });

        document.getElementById('overlay').style.display = 'none';
        document.getElementById('addMarkdown').textContent = '+';
    } catch (error) {
        console.error('Error processing file:', error);
        document.getElementById('progress-container').style.display = 'none';
        alert('Error processing file: ' + error.message);
    }
});

document.getElementById('addMarkdown').addEventListener('click', function() {
    const overlay = document.getElementById('overlay');
    if (overlay.style.display === 'none') {
        overlay.style.display = 'flex';
        this.textContent = '-';
    } else {
        overlay.style.display = 'none';
        this.textContent = '+';
    }
});

async function hasWebGPU() {
    if (!("gpu" in navigator)) {
        return 2;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            return 2;
        }
        if (adapter.features.has('shader-f16')) {
            return 0;
        }
        return 1;
    } catch (e) {
        console.error('WebGPU error:', e);
        return 2;
    }
}

async function Init(hasFP16) {
    try {
        if (kbContents.length === 0) {
            document.getElementById('chat-message').style.display = 'none';
        }

        rag = new RAG();
        await rag.InitPhi3SLM();
        await initializeWorker();
    } catch (error) {
        console.error('Initialization error:', error);
        alert('Error initializing application. Please refresh the page and try again.');
    }
}

window.onload = () => {
    hasWebGPU().then((supported) => {
        if (supported < 2) {
            if (supported == 1) {
                alert("Your GPU or Browser does not support webgpu with fp16, using fp32 instead.");
            }
            Init(supported === 0);
        } else {
            alert("Your GPU or Browser does not support webgpu");
        }
    }).catch(error => {
        console.error('Error checking WebGPU support:', error);
        alert("Error initializing WebGPU. Please check if your browser supports WebGPU.");
    });
};
