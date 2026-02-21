// FireGuard Firebase Configuration
// Project: fireguard-fbe22

const firebaseConfig = {
    apiKey: "AIzaSyAyyoXMZ8UDUd5SMk5g6cGloASpBvLDFBs",
    authDomain: "fireguard-fbe22.firebaseapp.com",
    projectId: "fireguard-fbe22",
    storageBucket: "fireguard-fbe22.firebasestorage.app",
    messagingSenderId: "191728247078",
    appId: "1:191728247078:web:59c9758802de2db9cee1e7",
    measurementId: "G-046VF9C0D8"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const messaging = firebase.messaging();

// Request Permission & Get Token
async function initializePushNotifications(swRegistration) {
    try {
        const permission = await Notification.requestPermission();
        if (permission !== 'granted') {
            console.log('Notification permission denied');
            return null;
        }

        console.log('Notification permission granted. Getting FCM token...');

        // Get FCM Token - pass the service worker registration
        const token = await messaging.getToken({
            vapidKey: 'BJ-Ev7bjBG-hqorEWWbV93Os5elWhdY182uac5S9Pua3bpd_2y1gUfbNVwVN6xrG7U7NDAD1KdzKFrYdI9noF6c',
            serviceWorkerRegistration: swRegistration
        });

        console.log('FCM Token:', token);

        // Send token to your Django backend to store it
        const deviceName = getDeviceName();
        await saveTokenToServer(token, deviceName);

        return token;
    } catch (error) {
        console.error('Error getting FCM token:', error);
        return null;
    }
}

// Get device name from browser info
function getDeviceName() {
    const ua = navigator.userAgent;
    let browser = 'Unknown Browser';
    let os = 'Unknown OS';

    // Detect browser
    if (ua.includes('Edg/')) browser = 'Edge';
    else if (ua.includes('Chrome/')) browser = 'Chrome';
    else if (ua.includes('Firefox/')) browser = 'Firefox';
    else if (ua.includes('Safari/')) browser = 'Safari';

    // Detect OS
    if (ua.includes('Windows')) os = 'Windows';
    else if (ua.includes('Android')) os = 'Android';
    else if (ua.includes('iPhone') || ua.includes('iPad')) os = 'iOS';
    else if (ua.includes('Mac OS')) os = 'macOS';
    else if (ua.includes('Linux')) os = 'Linux';

    return `${browser} on ${os}`;
}

// Save FCM token to Django backend
async function saveTokenToServer(token, deviceName) {
    try {
        const response = await fetch('/adminpanel/api/save-fcm-token/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({ token: token, device_name: deviceName })
        });

        if (response.ok) {
            const data = await response.json();
            console.log('FCM token saved to server:', data);
        } else {
            console.error('Failed to save FCM token. Status:', response.status);
        }
    } catch (error) {
        console.error('Error saving FCM token:', error);
    }
}

// Get CSRF token from cookies
function getCSRFToken() {
    const name = 'csrftoken';
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// ========== AUDIO BUZZER SYSTEM ==========
// Pre-create AudioContext and unlock it on first user interaction
let audioCtx = null;
let audioUnlocked = false;
let alarmInterval = null;

// Use BroadcastChannel to stop/start alarm in ALL tabs at once
const alarmChannel = new BroadcastChannel('fireguard_alarm');

alarmChannel.onmessage = (event) => {
    if (event.data === 'STOP_ALARM') {
        processStopAlarm(false); // Stop but don't broadcast back
    } else if (event.data === 'START_ALARM') {
        playAlarmSound(false);
    }
};

function getAudioContext() {
    if (!audioCtx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AC();
    }
    return audioCtx;
}

// Unlock audio on first user interaction (click/touch/keypress)
function unlockAudio() {
    if (audioUnlocked) return;
    try {
        const ctx = getAudioContext();
        if (ctx.state === 'suspended') {
            ctx.resume().then(() => {
                audioUnlocked = true;
                console.log('🔊 Audio unlocked - buzzer ready');
            });
        } else {
            audioUnlocked = true;
            console.log('🔊 Audio already unlocked');
        }
    } catch (e) {
        console.log('Audio unlock failed:', e);
    }
}

// Listen for user interactions to unlock audio
['click', 'touchstart', 'keydown'].forEach(event => {
    document.addEventListener(event, unlockAudio, { once: true });
});

// Handle foreground messages (when app is open)
messaging.onMessage((payload) => {
    console.log('Foreground message received:', payload);

    // Play alarm buzzer and notify other tabs
    playAlarmSound(true);

    // Show in-app notification with STOP button
    showInAppAlert(payload);
});

// Play alarm buzzer sound using Web Audio API
function playAlarmSound(shouldBroadcast = true) {
    try {
        const ctx = getAudioContext();

        // Resume if suspended (autoplay blocked)
        if (ctx.state === 'suspended') {
            ctx.resume();
        }

        // Stop any existing alarm
        if (alarmInterval) {
            clearInterval(alarmInterval);
        }

        let beepCount = 0;
        // removed maxBeeps to allow continuous ringing until stopped by user

        function playBeep() {
            const oscillator = ctx.createOscillator();
            const gainNode = ctx.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(ctx.destination);

            // Alternate between two VERY urgent frequencies (High-Low alarm)
            oscillator.frequency.value = beepCount % 2 === 0 ? 960 : 720;
            oscillator.type = 'square'; // Harsh square wave for better alertness
            gainNode.gain.value = 0.5;

            oscillator.start();
            oscillator.stop(ctx.currentTime + 0.2); // Longer beep
            beepCount++;
        }

        playBeep();
        alarmInterval = setInterval(playBeep, 400); // Continuous ringing
        console.log('🔊 CONTINUOUS Alarm buzzer playing!');

        if (shouldBroadcast) {
            alarmChannel.postMessage('START_ALARM');
        }

        // Show STOP button on dashboard if it exists
        const stopBtn = document.getElementById('stop-alarm-btn');
        if (stopBtn) stopBtn.classList.remove('hidden');

        return true;
    } catch (e) {
        console.log('Audio buzzer failed:', e);
        return false;
    }
}

// Stop the alarm buzzer
function stopAlarm() {
    processStopAlarm(true);
}

function processStopAlarm(shouldBroadcast = true) {
    if (alarmInterval) {
        clearInterval(alarmInterval);
        alarmInterval = null;
        console.log('🔕 Alarm stopped');
    }

    if (shouldBroadcast) {
        alarmChannel.postMessage('STOP_ALARM');
    }

    const el = document.getElementById('fcm-alert');
    if (el) el.remove();

    // Hide stop button on dashboard
    const stopBtn = document.getElementById('stop-alarm-btn');
    if (stopBtn) stopBtn.classList.add('hidden');
}

// Show in-app alert banner with alarm controls
function showInAppAlert(payload) {
    // Remove existing alert if any
    const existing = document.getElementById('fcm-alert');
    if (existing) existing.remove();

    const alertDiv = document.createElement('div');
    alertDiv.id = 'fcm-alert';
    alertDiv.innerHTML = `
        <div style="position: fixed; top: 0; left: 0; right: 0; background: linear-gradient(135deg, #dc2626, #f97316); color: white; padding: 15px 20px; z-index: 10000; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 20px rgba(0,0,0,0.5); animation: alertPulse 1s ease-in-out infinite;">
            <div style="display: flex; align-items: center; gap: 12px; max-width: 60%;">
                <span style="font-size: 24px;">🔥</span>
                <div style="overflow: hidden;">
                    <strong style="font-size: 14px; display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${payload.notification?.title || 'Fire Alert!'}</strong>
                    <p style="margin: 0; font-size: 12px; opacity: 0.9; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${payload.notification?.body || 'Check dashboard immediately.'}</p>
                </div>
            </div>
            <div style="display: flex; gap: 5px;">
                <button onclick="stopAlarm()" style="background: white; border: none; color: #dc2626; padding: 10px 15px; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 14px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">STOP ALARM</button>
            </div>
        </div>
    `;

    // Add pulse animation
    if (!document.getElementById('fcm-alert-style')) {
        const style = document.createElement('style');
        style.id = 'fcm-alert-style';
        style.textContent = `
            @keyframes alertPulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.85; }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.prepend(alertDiv);

    // Auto-remove after 5 minutes (extremely persistent)
    setTimeout(() => {
        stopAlarm();
    }, 300000);
}

// Helper: wait for service worker to become active
function waitForSWActive(registration) {
    return new Promise((resolve) => {
        if (registration.active) {
            resolve(registration);
            return;
        }
        const sw = registration.installing || registration.waiting;
        if (sw) {
            sw.addEventListener('statechange', () => {
                if (sw.state === 'activated') {
                    resolve(registration);
                }
            });
        } else {
            resolve(registration);
        }
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    if ('serviceWorker' in navigator) {
        // Register the Firebase messaging service worker
        navigator.serviceWorker.register('/firebase-messaging-sw.js')
            .then((registration) => {
                console.log('ServiceWorker registered:', registration.scope);

                // Wait for service worker to become active before getting FCM token
                return waitForSWActive(registration);
            })
            .then((activeRegistration) => {
                console.log('ServiceWorker is active. Initializing FCM...');
                initializePushNotifications(activeRegistration);
            })
            .catch((error) => {
                console.error('ServiceWorker registration failed:', error);
            });

        // Listen for messages from service worker (when user taps notification)
        navigator.serviceWorker.addEventListener('message', (event) => {
            if (event.data && event.data.type === 'FIRE_ALERT') {
                console.log('🔥 Fire alert received from service worker!');
                playAlarmSound();
                showInAppAlert({
                    notification: {
                        title: event.data.title,
                        body: event.data.body
                    }
                });
            }
        });
    }

    // Auto-play alarm if page was opened from a notification click (?alarm=true)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('alarm') === 'true') {
        console.log('🔥 Page opened from notification - playing alarm!');
        // Small delay to ensure AudioContext is ready
        setTimeout(() => {
            playAlarmSound();
            showInAppAlert({
                notification: {
                    title: '🔥 Fire Alert!',
                    body: 'Fire/Smoke detected! Check the dashboard immediately!'
                }
            });
        }, 500);
        // Clean up URL
        window.history.replaceState({}, '', window.location.pathname);
    }
});
