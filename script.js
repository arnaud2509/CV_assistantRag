import { Application } from "https://unpkg.com/@splinetool/runtime@1.6.3/build/runtime.js";

const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const responseDisplay = document.getElementById('response-display');
const voiceWave = document.getElementById('voice-wave');
const canvas3d = document.getElementById('canvas3d');

let splineApp, currentAudio = null;

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

// --- FONCTION PRINCIPALE : Appel API et lecture audio ---
async function sendMessage() {
  const query = userInput.value.trim();
  if (!query) return;

  sendButton.disabled = true;
  userInput.disabled = true;
  showLoading();
  userInput.value = '';

  try {
    // 1. APPEL AU BACKEND
    const response = await fetch(`${RENDER_API_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: query })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({detail: "Réponse non JSON"}));
      throw new Error(`Erreur API: ${response.status} - ${errorData.detail || 'Erreur inconnue'}`);
    }

    // 2. Récupération du Blob audio
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);

    // 3. Typing effect et lecture audio en parallèle
    voiceWave.classList.remove('hidden');
    const typingPromise = typeWriterEffect("..."); // Placeholder texte si tu veux afficher un "..." pendant lecture
    const audioPromise = new Promise(resolve => {
      audio.onended = () => {
        voiceWave.classList.add('hidden');
        URL.revokeObjectURL(audioUrl);
        resolve();
      };
      audio.play().catch(e => {
        console.error("Erreur lecture audio :", e);
        voiceWave.classList.add('hidden');
        resolve();
      });
    });

    await Promise.all([typingPromise, audioPromise]);

  } catch (err) {
    handleError(err, "Échec de la communication avec l'API RAG. Vérifiez la console pour les détails.");
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

// Expose sendMessage au scope global
window.sendMessage = sendMessage;

console.log("Canvas trouvé :", canvas3d);
window.onload = () => userInput.focus();
