import { RAG } from './rag.js';

let rag;

// Initialize RAG in the worker
async function initRag() {
  rag = new RAG();
  await rag.InitPhi3SLM();
  self.postMessage({ type: 'initialized' });
}

self.onmessage = async function(e) {
  if (e.data.type === 'init') {
    await initRag();
  } else if (e.data.type === 'process') {
    const content = e.data.content;
    try {
      const prompt_template = `
        Extract the input according to the following conditions, keep the knowledge points, summarize the detailed content of the knowledge points and delete the content related to the picture before outputting, please keep the content related to the link when outputting

        Starting with # , ## , ### is a knowledge point, All knowledge points must be output and cannot be missing, such as

        [INPUT]
        # ABC
        ...............

        ## CDF 
        ..........
        ### GGG
        .....
        [END INPUT]

        [OUTPUT]
        [{"KB": "ABC", "Content":"..............."},{"KB": "CDF", "Content":".........."},{"KB": "GGG", "Content":"....."}]
        [END OUTPUT]

        [INPUT]
        # ABC
        ...............

        ### GGG
        .....

        ### www
        .....

        [END INPUT]

        [OUTPUT]
        [{"KB": "ABC", "Content":"..............."},{"KB": "GGG", "Content":"....."},{"KB": "www", "Content":"....."}]
        [END OUTPUT]


        [INPUT]
      ` + content + 
      `
        [END INPUT]

       [OUTPUT]`;

      const prompt = `<|system|>\nYou are a markdown assistant. Help me to get knowledge in markdown file<|end|>\n<|user|>\n${prompt_template}<|end|>\n<|assistant|>\n`;

      // Process in chunks to avoid blocking
      const json_result = await rag.generateEmbeddingsContent(prompt);
      const jsonObject = JSON.parse(json_result);

      const kbContents = [];
      for(const content of jsonObject) {
        kbContents.push(content.KB + '-' + content.Content);
        // Send progress updates
        self.postMessage({ 
          type: 'progress', 
          progress: (kbContents.length / jsonObject.length) * 100 
        });
      }

      self.postMessage({ 
        type: 'complete', 
        result: kbContents 
      });

    } catch (error) {
      self.postMessage({ 
        type: 'error', 
        error: error.message 
      });
    }
  }
};
