import { Application } from "https://unpkg.com/@splinetool/runtime@1.6.3/build/runtime.js";

const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const responseDisplay = document.getElementById('response-display');
const voiceWave = document.getElementById('voice-wave');
const canvas3d = document.getElementById('canvas3d');

let splineApp, currentAudio = null;

// --- CONFIGURATION DE L'API RAG ---
const RENDER_API_URL = "https://cv-assistantrag.onrender.com"; 
// **********************************

// === Initialisation Spline (Conserver tel quel) ===
const observer = new IntersectionObserver(entries => {
  if (entries[0].isIntersecting) {
    splineApp = new Application(canvas3d);
    splineApp.load('https://prod.spline.design/Jpxk6lxmkWLWUPls/scene.splinecode')
      .then(() => console.log("Scène Spline 3D chargée."));
    observer.disconnect();
  }
});
observer.observe(document.getElementById('spline-wrapper'));

// === UTILITAIRES AUDIO ===

// Convertit Base64 en ArrayBuffer
function base64ToArrayBuffer(base64) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

// Fonction utilitaire pour écrire des chaînes dans la vue binaire
function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

// Convertit les données PCM 16 bits (raw) en fichier WAV (nécessaire pour la lecture dans le navigateur)
function pcmToWav(pcm16, sampleRate = 24000) { // On suppose un sample rate de 24000
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcm16.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF Chunk
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');
  
  // FMT Chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // Size of the chunk
  view.setUint16(20, 1, true);  // Audio format 1=PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // Bits per sample
  
  // Data Chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);
  
  // Write PCM Data
  let offset = 44;
  for (let i = 0; i < pcm16.length; i++, offset += 2) {
    view.setInt16(offset, pcm16[i], true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}


// === UTILITAIRES UI ===
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
  responseDisplay.style.textAlign = 'left'; // Alignement standard pour la réponse
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
  voiceWave.classList.add('hidden'); // S'assurer que l'animation est masquée
}


// --- FONCTION PRINCIPALE : Appel à l'API RAG ---
async function callRagApi(query) {
  const apiUrl = `${RENDER_API_URL}/ask`; 

  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question: query }), 
  });

  if (!response.ok) {
    // Gérer les erreurs HTTP (4xx, 5xx)
    const errorData = await response.json().catch(() => ({detail: "Réponse non JSON"}));
    throw new Error(`Erreur API: ${response.status} - ${errorData.detail || 'Erreur inconnue'}`);
  }

  const data = await response.json();
  // L'API FastAPI retourne {"answer": "..."}
  if (data && data.answer) {
      return data.answer; 
  }
  throw new Error("Réponse de l'API incomplète ou manquante.");
}


// === FONCTION D'ENVOI DE MESSAGE UNIFIÉE ===
async function sendMessage() {
  const query = userInput.value.trim();
  if (!query) return;

  // Désactiver les contrôles et afficher le chargement
  sendButton.disabled = true;
  userInput.disabled = true;
  showLoading();
  userInput.value = '';

  let audioUrlToRevoke = null; // Pour gérer l'URL Blob audio
  
  try {
    // 1. APPEL AU BACKEND RÉEL
    const responseText = await callRagApi(query); 

    // 2. Traitement de la réponse (Recherche de l'audio TTS)
    // Le backend RAG ne génère pas de TTS par défaut, donc nous traitons la réponse TEXTUELLE seulement.
    let finalResponseText = responseText;
    let audio = null; 
    
    // Si votre backend était capable de retourner du TTS encodé en Base64 :
    // const audioMatch = responseText.match(/\[AUDIO_BASE64:(.*?)\]/);
    // if (audioMatch) {
    //     const base64Audio = audioMatch[1];
    //     finalResponseText = responseText.replace(audioMatch[0], '').trim();
    //     const pcmData = base64ToArrayBuffer(base64Audio);
    //     const pcm16 = new Int16Array(pcmData);
    //     const wavBlob = pcmToWav(pcm16);
    //     audioUrlToRevoke = URL.createObjectURL(wavBlob);
    //     audio = new Audio(audioUrlToRevoke);
    // }

    // 3. Afficher la réponse (Typing Effect)
    const typingPromise = typeWriterEffect(finalResponseText);
    let audioPlayPromise = Promise.resolve();

    if (audio) {
      audioPlayPromise = new Promise(resolve => {
        voiceWave.classList.remove('hidden');

        audio.onended = () => {
          voiceWave.classList.add('hidden');
          resolve();
        };
        
        audio.play().catch(e => {
          console.error("Échec de la lecture audio, résolution immédiate:", e);
          voiceWave.classList.add('hidden');
          resolve();
        });
      });
    }
    
    // 4. Attendre que le typing et la lecture audio (si existante) soient terminés
    await Promise.all([typingPromise, audioPlayPromise]);

    if (splineApp) console.log("Interaction Spline possible :", query);

  } catch (err) {
    // Gérer les erreurs de l'API ou du réseau
    handleError(err, "Échec de la communication avec l'API RAG. Vérifiez la console pour les détails.");
  } finally {
      // 5. Nettoyage et réactivation
      if (audioUrlToRevoke) {
          URL.revokeObjectURL(audioUrlToRevoke);
      }
    sendButton.disabled = false;
    userInput.disabled = false;
    userInput.focus();
    hideLoading();
  }
}

// Fonction pour masquer le chargement une fois la réponse terminée
function hideLoading() {
    // Si l'affichage contient encore l'animation de chargement, on l'efface.
    if (responseDisplay.querySelector('.loading-dot')) {
        responseDisplay.innerHTML = '';
    }
}


// Expose sendMessage au scope global pour les attributs HTML (onclick/onkeydown)
window.sendMessage = sendMessage;

// Configuration initiale
console.log("Canvas trouvé :", document.getElementById('canvas3d'));
window.onload = () => userInput.focus();
