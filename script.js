import { Application } from "https://unpkg.com/@splinetool/runtime@1.6.3/build/runtime.js";

const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const responseDisplay = document.getElementById('response-display');
const voiceWave = document.getElementById('voice-wave');
const canvas3d = document.getElementById('canvas3d');

// === VARIABLES GLOBALES ===
let splineApp = null;
let currentAudio = null; // Variable globale pour l'objet Audio
const RENDER_API_URL = "https://cv-assistantrag.onrender.com";

// === Initialisation Spline (inchangée) ===
const observer = new IntersectionObserver(entries => {
  if (entries[0].isIntersecting) {
    splineApp = new Application(canvas3d);
    splineApp.load('https://prod.spline.design/Jpxk6lxmkWLWUPls/scene.splinecode')
      .then(() => console.log("Scène Spline 3D chargée."));
    observer.disconnect();
  }
});
observer.observe(document.getElementById('spline-wrapper'));

// === UTILITAIRES DE CONVERSION AUDIO (Nécessaires pour l'API Gemini TTS/PCM) ===

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
  const bytesPerSample = 2; // Int16Array
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcm16.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF chunk
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // fmt sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // Audio format (1 for PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // Bits per sample (16-bit)

  // data sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Write PCM data
  let offset = 44;
  for (let i = 0; i < pcm16.length; i++, offset += 2) {
    view.setInt16(offset, pcm16[i], true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/** Arrête et nettoie l'objet audio en cours. */
function stopAudio() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.onended = null;
    // Révoque l'URL pour libérer la mémoire du Blob
    if (currentAudio.src) {
      URL.revokeObjectURL(currentAudio.src);
    }
    currentAudio = null;
  }
  voiceWave.classList.add('hidden');
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

function hideLoading() {
  if (responseDisplay.querySelector('.loading-dot') && responseDisplay.innerHTML.trim() === '') {
    responseDisplay.innerHTML = '';
  }
}

function typeWriterEffect(text) {
  let i = 0;
  responseDisplay.innerHTML = '';
  responseDisplay.style.textAlign = 'left';
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
  stopAudio(); // S'assure que l'audio s'arrête en cas d'erreur
}


// --- FONCTION PRINCIPALE : Appel à l'API RAG (Récupère seulement le texte) ---
async function callRagApi(query) {
  const apiUrl = `${RENDER_API_URL}/ask`;
  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    // La question est envoyée au backend RAG
    body: JSON.stringify({ question: query })
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({detail: "Réponse non JSON"}));
    throw new Error(`Erreur API: ${response.status} - ${errorData.detail || 'Erreur inconnue'}`);
  }

  const data = await response.json();
  if (data && data.answer) {
    // On retourne uniquement la réponse textuelle (answer)
    return data.answer; 
  }
  throw new Error("Réponse de l'API incomplète ou manquante.");
}


// --- FONCTION TTS : Génération et lecture de la voix (Client-side) ---
async function prepareAudio(text) {
  stopAudio(); // Arrête tout audio en cours
  voiceWave.classList.remove('hidden'); // Affiche l'onde sonore pendant le chargement/lecture

  const apiKey = ""; // La clé sera fournie par l'environnement
  const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key=${apiKey}`;

  const payload = {
    contents: [{
      parts: [{ text: text }]
    }],
    generationConfig: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: {
          // Utilisation de la voix "Kore" pour un ton ferme/clair
          prebuiltVoiceConfig: { voiceName: "Kore" }
        }
      }
    },
    model: "gemini-2.5-flash-preview-tts"
  };

  const maxRetries = 5;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`API TTS a retourné un statut ${response.status}`);
      }

      const result = await response.json();
      const part = result?.candidates?.[0]?.content?.parts?.[0];
      const audioData = part?.inlineData?.data;
      const mimeType = part?.inlineData?.mimeType;

      if (audioData && mimeType && mimeType.startsWith("audio/L16")) {
        // 1. Extraire le taux d'échantillonnage et les données PCM
        const rateMatch = mimeType.match(/rate=(\d+)/);
        const sampleRate = rateMatch ? parseInt(rateMatch[1], 10) : 24000;
        
        const pcmData = base64ToArrayBuffer(audioData);
        const pcm16 = new Int16Array(pcmData);

        // 2. Convertir en Blob WAV et créer l'URL
        const wavBlob = pcmToWav(pcm16, sampleRate);
        const audioUrl = URL.createObjectURL(wavBlob);

        // 3. Jouer l'audio
        const audio = new Audio(audioUrl);
        currentAudio = audio; // Met à jour la variable globale

        audio.onended = () => stopAudio(); // S'arrête et nettoie à la fin
        audio.onerror = () => stopAudio();
        
        // Tente de jouer l'audio
        audio.play().catch(e => {
          stopAudio();
          console.error("Erreur de lecture audio (Autoplay bloqué?):", e);
        });

        return; // Succès, sortie de la fonction
      } else {
        throw new Error("Réponse TTS manquante ou format audio incorrect.");
      }
    } catch (error) {
      console.warn(`Tentative ${attempt + 1} échouée pour TTS: ${error.message}. Réessaie...`);
      if (attempt === maxRetries - 1) {
        throw new Error(`Échec de la TTS après ${maxRetries} tentatives.`);
      }
      // Délai de backoff exponentiel (1s, 2s, 4s, etc.)
      const delay = Math.pow(2, attempt) * 1000;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}


// === FONCTION D'ENVOI DE MESSAGE UNIFIÉE ===
async function sendMessage() {
  const query = userInput.value.trim();
  if (!query) return;

  // Arrêter l'audio précédent
  stopAudio();

  sendButton.disabled = true;
  userInput.disabled = true;
  showLoading();
  userInput.value = '';

  try {
    // 1. Appel à l'API RAG pour obtenir la réponse TEXTUELLE
    const answerText = await callRagApi(query);

    // 2. Affichage texte (effet machine à écrire)
    const typingPromise = typeWriterEffect(answerText);
    await typingPromise; 

    // 3. Génération et lecture audio client-side
    const audioPromise = prepareAudio(answerText);
    await audioPromise; 
    
    if (splineApp) console.log("Interaction Spline possible :", query);

  } catch (err) {
    // En cas d'erreur RAG ou TTS, tout est géré ici
    handleError(err, "Échec de la communication. Le service RAG ou la synthèse vocale a échoué.");
  } finally {
    sendButton.disabled = false;
    userInput.disabled = false;
    userInput.focus();
    hideLoading();
  }
}

window.sendMessage = sendMessage;
window.onload = () => userInput.focus();
