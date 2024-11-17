import { env, AutoTokenizer } from '@xenova/transformers';
import { LLM } from './llm.js';
import { marked } from 'marked';
import { ragSystem } from './rag.js';
import { clipboardIcon } from './icons.js';

const MODELS = {
  "phi3": { name: "phi3", path: "microsoft/Phi-3-mini-4k-instruct-onnx-web", externaldata: true },
  "phi3dev": { name: "phi3dev", path: "schmuell/Phi-3-mini-4k-instruct-onnx-web", externaldata: true },
}

function updateStatus(message) {
  const statusElement = document.getElementById('status');
  statusElement.textContent = message;
  console.log(message);
}

function enableInterface() {
  document.getElementById('user-input').disabled = false;
  document.getElementById('send-button').disabled = false;
  document.getElementById('status-container').style.display = 'none';
  document.getElementById('knowledge-input').disabled = false;
  document.getElementById('process-text').disabled = false;
  document.getElementById('process-pdf').disabled = false;
  document.getElementById('pdf-upload').disabled = false;
  document.getElementById('user-input').focus();
}

marked.use({ mangle: false, headerIds: false });

const sendButton = document.getElementById('send-button');
const scrollWrapper = document.getElementById('scroll-wrapper');
const userInput = document.getElementById('user-input');
const knowledgeInput = document.getElementById('knowledge-input');
const processTextButton = document.getElementById('process-text');
const pdfUpload = document.getElementById('pdf-upload');
const processPdfButton = document.getElementById('process-pdf');
const processStatus = document.getElementById('process-status');

// Handle knowledge base text processing
processTextButton.addEventListener('click', async () => {
  const text = knowledgeInput.value.trim();
  if (text.length === 0) {
    processStatus.textContent = 'Please enter some text to process.';
    return;
  }

  processStatus.textContent = 'Processing text...';
  processTextButton.disabled = true;
  knowledgeInput.disabled = true;

  try {
    // Clear previous knowledge base
    ragSystem.clear();
    
    // Process new text
    const numChunks = await ragSystem.processText(text);
    processStatus.textContent = `Successfully processed text into ${numChunks} chunks.`;
  } catch (error) {
    console.error(error);
    processStatus.textContent = 'Error processing text. Please try again.';
  } finally {
    processTextButton.disabled = false;
    knowledgeInput.disabled = false;
  }
});

// Handle PDF processing
processPdfButton.addEventListener('click', async () => {
  const file = pdfUpload.files[0];
  if (!file) {
    processStatus.textContent = 'Please select a PDF file.';
    return;
  }

  if (file.type !== 'application/pdf') {
    processStatus.textContent = 'Please upload a valid PDF file.';
    return;
  }

  processStatus.textContent = 'Processing PDF...';
  processPdfButton.disabled = true;
  pdfUpload.disabled = true;

  try {
    // Clear previous knowledge base
    ragSystem.clear();
    
    // Process PDF
    const numChunks = await ragSystem.processPDF(file);
    processStatus.textContent = `Successfully processed PDF into ${numChunks} chunks.`;
  } catch (error) {
    console.error(error);
    processStatus.textContent = 'Error processing PDF. Please try again.';
  } finally {
    processPdfButton.disabled = false;
    pdfUpload.disabled = false;
  }
});

//
// auto scroll the content area until a user scrolls up
//
let isAutoScrollOn = true;
let lastKnownScrollPosition = 0;
let ticking = false;

const autoScroller = new ResizeObserver(() => {
  if (isAutoScrollOn) {
    scrollWrapper.scrollIntoView({ behavior: "smooth", block: "end" });
  }
});

document.addEventListener("scroll", () => {
  if (!ticking && isAutoScrollOn && window.scrollY < lastKnownScrollPosition) {
    window.requestAnimationFrame(() => {
      isAutoScrollOn = false;
      ticking = false;
    });
    ticking = true;
  }
  else if (!ticking && !isAutoScrollOn && window.scrollY > lastKnownScrollPosition &&
    window.scrollY >= document.documentElement.scrollHeight - window.innerHeight - 30) {
    window.requestAnimationFrame(() => {
      isAutoScrollOn = true;
      ticking = false;
    });
    ticking = true;
  }
  lastKnownScrollPosition = window.scrollY;
});

//
// make response available for copying to clipboard
//
function copyTextToClipboard(responseDiv) {
  const copyButton = document.createElement('button');
  copyButton.className = 'btn btn-secondary copy-button';
  copyButton.innerHTML = clipboardIcon;
  copyButton.onclick = () => {
    navigator.clipboard.writeText(responseDiv.innerText);
  };
  responseDiv.appendChild(copyButton);
}

// 
// user hits send, enter or ctl enter
//
async function submitRequest(e) {
  if (sendButton.innerHTML == "Stop") {
    llm.abort();
    return;
  }

  // enter clears the chat history, ctl enter will continue the conversation
  const continuation = e && e.ctrlKey && e.key === 'Enter';

  document.getElementById('chat-container').style.display = 'block';

  let input = userInput.value.trim();
  if (input.length == 0) {
    return;
  }

  let context = document.getElementById('chat-history').context;
  if (context === undefined) {
    context = "";
  }

  // append to chat history
  let chatHistory = document.getElementById('chat-history');
  let userMessageDiv = document.createElement('div');
  userMessageDiv.className = 'user-message';
  userMessageDiv.innerText = input;
  chatHistory.appendChild(userMessageDiv);

  // container for llm response
  let responseDiv = document.createElement('div');
  responseDiv.className = 'response-message text-start';
  responseDiv.style.minHeight = '3em';
  let spinner = document.createElement('div');
  spinner.className = 'spinner-border text-dark';
  spinner.setAttribute('role', 'status');
  responseDiv.appendChild(spinner);
  chatHistory.appendChild(responseDiv);

  // toggle button to stop text generation
  sendButton.innerHTML = "Stop";

  // change autoScroller to keep track of our new responseDiv
  autoScroller.observe(responseDiv);

  if (continuation) {
    input = context + " " + input;
  }

  try {
    await Query(continuation, input, (word) => {
      responseDiv.innerHTML = marked.parse(word);
    });
    chatHistory.context = responseDiv.innerHTML;
    copyTextToClipboard(responseDiv);
  } catch (error) {
    console.error(error);
    responseDiv.innerHTML = "Sorry, there was an error processing your request. Please try again.";
  } finally {
    sendButton.innerHTML = "Send";
    spinner.remove();
    userInput.value = '';
    userInput.focus();
  }
}

// 
// event listener for Ctrl+Enter or Enter
//
userInput.addEventListener('keydown', function (e) {
  if (e.key === 'Enter') {
    e.preventDefault();
    if (e.ctrlKey) {
      submitRequest(e);
    } else if (!e.shiftKey) {
      submitRequest(e);
    }
  }
});

// Click handler for send button
sendButton.addEventListener('click', () => submitRequest());

function getConfig() {
  const query = window.location.search.substring(1);
  var config = {
    model: "phi3",
    provider: "webgpu",
    profiler: 0,
    verbose: 0,
    threads: 1,
    show_special: 0,
    csv: 0,
    max_tokens: 9999,
    local: 0,
  }
  let vars = query.split("&");
  for (var i = 0; i < vars.length; i++) {
    let pair = vars[i].split("=");
    if (pair[0] in config) {
      const key = pair[0];
      const value = decodeURIComponent(pair[1]);
      if (typeof config[key] == "number") {
        config[key] = parseInt(value);
      }
      else {
        config[key] = value;
      }
    } else if (pair[0].length > 0) {
      throw new Error("unknown argument: " + pair[0]);
    }
  }
  if (MODELS[config.model] !== undefined) {
    config.model = MODELS[config.model];
  }
  return config;
}

const config = getConfig();

// setup for transformers.js tokenizer
env.localModelPath = 'models';
env.allowRemoteModels = config.local == 0;
env.allowLocalModels = config.local == 1;

let tokenizer;
const llm = new LLM();

function token_to_text(tokenizer, tokens, startidx) {
  const txt = tokenizer.decode(tokens.slice(startidx), { skip_special_tokens: config.show_special != 1, });
  return txt;
}

async function Query(continuation, query, cb) {
  // Get relevant context from RAG system if available
  let relevantContext = '';
  if (ragSystem.embeddings.length > 0) {
    const contexts = await ragSystem.findRelevantContext(query);
    relevantContext = contexts.join('\n\n');
  }

  let prompt = (continuation) ? query : 
    `<|system|>\nYou are a friendly assistant. ${relevantContext ? 'Use the following context to help answer the question:\n\n' + relevantContext + '\n\n' : ''}<|end|>\n<|user|>\n${query}<|end|>\n<|assistant|>\n`;

  const { input_ids } = await tokenizer(prompt, { return_tensor: false, padding: true, truncation: true });

  // clear caches 
  llm.initilize_feed();

  const start_timer = performance.now();
  const output_index = llm.output_tokens.length + input_ids.length;
  const output_tokens = await llm.generate(input_ids, (output_tokens) => {
    if (output_tokens.length == input_ids.length + 1) {
      // time to first token
      const took = (performance.now() - start_timer) / 1000;
      console.log(`time to first token in ${took.toFixed(1)}sec, ${input_ids.length} tokens`);
    }
    cb(token_to_text(tokenizer, output_tokens, output_index));
  }, { max_tokens: config.max_tokens });

  const took = (performance.now() - start_timer) / 1000;
  cb(token_to_text(tokenizer, output_tokens, output_index));
  const seqlen = output_tokens.length - output_index;
  console.log(`${seqlen} tokens in ${took.toFixed(1)}sec, ${(seqlen / took).toFixed(2)} tokens/sec`);
}

//
// Load the model and tokenizer
//
async function Init(hasFP16) {
  try {
    updateStatus("Loading tokenizer...");
    tokenizer = await AutoTokenizer.from_pretrained(config.model.path);

    updateStatus("Loading model...");
    await llm.load(config.model, {
      provider: config.provider,
      profiler: config.profiler,
      verbose: config.verbose,
      local: config.local,
      max_tokens: config.max_tokens,
      hasFP16: hasFP16,
    });

    updateStatus("Initializing RAG system...");
    await ragSystem.initialize();
    
    updateStatus("Ready! You can now start chatting.");
    enableInterface();
  } catch (error) {
    updateStatus("Error: " + error.message);
    console.error(error);
  }
}

//
// Check if we have webgpu and fp16
//
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

window.onload = () => {
  hasWebGPU().then((supported) => {
    if (supported < 2) {
      if (supported == 1) {
        updateStatus("Your GPU or Browser does not support webgpu with fp16, using fp32 instead.");
      }
      Init(supported === 0);
    } else {
      updateStatus("Error: Your GPU or Browser does not support webgpu");
    }
  });
}
