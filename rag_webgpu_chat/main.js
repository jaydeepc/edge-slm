import { env, AutoTokenizer,pipeline, cos_sim } from '@xenova/transformers';
import { RAG } from './rag.js';
import { Phi3SLM } from './phi3_slm.js';
import { marked } from 'marked';

const fileContents = [];
const kbContents = [];
let rag;
let embeddingWorker;

document.getElementById('send-button').addEventListener('click', async function() {
  const inputBox = document.querySelector('.input-box');
  const question = inputBox.value.trim();
  
  if (question) {
      const chatContainer = document.querySelector('.chat-container');
      const userMessage = document.createElement('div');
      userMessage.className = 'message question';
      userMessage.textContent = question;
      chatContainer.appendChild(userMessage);

      inputBox.value = '';

      try {
          const answer = await generateAnswer(question);
          const summary = await generativeSummary(answer);

          const botMessage = document.createElement('div');
          botMessage.className = 'message answer';
          botMessage.textContent = summary;
          chatContainer.appendChild(botMessage);

          chatContainer.scrollTop = chatContainer.scrollHeight;
      } catch (error) {
          console.error('Error fetching answer:', error);
      }
  }
});

async function generateAnswer(question) {
  return rag.getEmbeddings(question, kbContents);
}

async function generativeSummary(answer) {
  let prompt = `<|system|>\nYou are a friendly assistant. Help me summarize answer of the knowledge points<|end|>\n<|user|>\n${answer}<|end|>\n<|assistant|>\n`;
  let summary = await rag.generateSummaryContent(prompt);
  return summary;
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

  for (const file of files) {
      const text = await file.text();
      fileContents.push(text);
  }

  document.getElementById('progress-container').style.display = 'block';

  // Send content to worker for processing
  embeddingWorker.postMessage({ 
    type: 'process', 
    content: fileContents[0] 
  });

  document.getElementById('overlay').style.display = 'none';
  document.getElementById('addMarkdown').textContent = '+';
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
    const adapter = await navigator.gpu.requestAdapter()
    if (adapter.features.has('shader-f16')) {
      return 0;
    }
    return 1;
  } catch (e) {
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
    console.log('InitError');
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
  });
}
