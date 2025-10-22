import { Application } from "https://unpkg.com/@splinetool/runtime@1.6.3/build/runtime.js";

const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const responseDisplay = document.getElementById('response-display');
const voiceWave = document.getElementById('voice-wave');
const canvas3d = document.getElementById('canvas3d');

let splineApp = null;

// --- CONFIGURATION DE L'API RAG ---
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
  voiceWave.classList.add('hidden');
}

// --- FONCTION PRINCIPALE : Appel à l'API RAG ---
async function callRagApi(query) {
  const apiUrl = `${RENDER_API_URL}/ask`;
  const response = await fetch(apiUrl, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ question: query })
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({detail: "Réponse non JSON"}));
    throw new Error(`Erreur API: ${response.status} - ${errorData.detail || 'Erreur inconnue'}`);
  }

  const data = await response.json();
  if (data && data.answer) {
    return data;
  }
  throw new Error("Réponse de l'API incomplète ou manquante.");
}

// === FONCTION D'ENVOI DE MESSAGE UNIFIÉE ===
async function sendMessage() {
  const query = userInput.value.trim();
  if (!query) return;

  sendButton.disabled = true;
  userInput.disabled = true;
  showLoading();
  userInput.value = '';

  let currentAudio = null;

  try {
    const { answer, audio_base64 } = await callRagApi(query);

    // --- Affichage texte ---
    const typingPromise = typeWriterEffect(answer);

    // --- Lecture audio si présent ---
    let audioPlayPromise = Promise.resolve();
    if (audio_base64) {
      const audioBlob = new Blob([Uint8Array.from(atob(audio_base64), c => c.charCodeAt(0))], { type: 'audio/mp3' });
      const audioUrl = URL.createObjectURL(audioBlob);
      currentAudio = new Audio(audioUrl);

      audioPlayPromise = new Promise(resolve => {
        voiceWave.classList.remove('hidden');
        currentAudio.onended = () => {
          voiceWave.classList.add('hidden');
          URL.revokeObjectURL(audioUrl);
          resolve();
        };
        currentAudio.play().catch(e => {
          console.error("Échec lecture audio:", e);
          voiceWave.classList.add('hidden');
          resolve();
        });
      });
    }

    // --- Attendre texte + audio ---
    await Promise.all([typingPromise, audioPlayPromise]);

    if (splineApp) console.log("Interaction Spline possible :", query);

  } catch (err) {
    handleError(err, "Échec de la communication avec l'API RAG. Vérifiez la console.");
  } finally {
    sendButton.disabled = false;
    userInput.disabled = false;
    userInput.focus();
    hideLoading();
  }
}

function hideLoading() {
  if (responseDisplay.querySelector('.loading-dot')) {
    responseDisplay.innerHTML = '';
  }
}

window.sendMessage = sendMessage;
window.onload = () => userInput.focus();
