import { Application } from "https://unpkg.com/@splinetool/runtime@1.6.3/build/runtime.js";

const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const responseDisplay = document.getElementById('response-display');
const voiceWave = document.getElementById('voice-wave');
const canvas3d = document.getElementById('canvas3d');

let splineApp, currentAudio = null;

// === Initialisation Spline ===
const observer = new IntersectionObserver(entries => {
  if (entries[0].isIntersecting) {
    splineApp = new Application(canvas3d);
    splineApp.load('https://prod.spline.design/Jpxk6lxmkWLWUPls/scene.splinecode')
      .then(() => console.log("Scène Spline 3D chargée."));
    observer.disconnect();
  }
});
observer.observe(document.getElementById('spline-wrapper'));

// === UTILITAIRES ===
function base64ToArrayBuffer(base64) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function pcmToWav(pcm16, sampleRate) {
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcm16.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < pcm16.length; i++, offset += 2) {
    view.setInt16(offset, pcm16[i], true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function showLoading() {
  responseDisplay.style.opacity = 0;
  responseDisplay.innerHTML = `
    <div class="flex justify-center items-center space-x-2 w-full h-full">
      <span class="loading-dot"></span>
      <span class="loading-dot"></span>
      <span class="loading-dot"></span>
    </div>`;
  responseDisplay.style.opacity = 1;
}

function typeWriterEffect(text) {
  let i = 0;
  responseDisplay.innerHTML = '';
  responseDisplay.style.textAlign = 'center';
  return new Promise(resolve => {
    const speed = 40;
    const interval = setInterval(() => {
      responseDisplay.innerHTML = text.slice(0, ++i);
      if (i >= text.length) {
        clearInterval(interval);
        resolve();
      }
    }, speed);
  });
}

function handleError(error, userMessage = "Une erreur est survenue. Veuillez réessayer.") {
  console.error(error);
  responseDisplay.innerHTML = userMessage;
  responseDisplay.style.opacity = 1;
}

// === BACKEND SIMULÉ ===
function simulateBackendResponse(query) {
  return new Promise(resolve => {
    setTimeout(() => {
      let response = `(Réponse simulée) Pour "${query}"`;
      if (query.toLowerCase().includes('sap')) {
        response = "Arnaud maîtrise SAP FI/CO et a simplifié l'expérience utilisateur de Fiori.";
      }
      resolve(response);
    }, 1500);
  });
}

// === MAIN ===
async function sendMessage() {
  const query = userInput.value.trim();
  if (!query) return;
  sendButton.disabled = true;
  userInput.disabled = true;
  showLoading();
  userInput.value = '';

  try {
    const responseText = await simulateBackendResponse(query);
    await typeWriterEffect(responseText);
  } catch (err) {
    handleError(err);
  } finally {
    sendButton.disabled = false;
    userInput.disabled = false;
    userInput.focus();
  }
}

console.log("Canvas trouvé :", document.getElementById('canvas3d'));
window.sendMessage = sendMessage;
window.onload = () => userInput.focus();
