// Firebase Messaging Service Worker (Required by FCM)
// This file MUST be served at /firebase-messaging-sw.js

importScripts('https://www.gstatic.com/firebasejs/9.22.0/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/9.22.0/firebase-messaging-compat.js');

// Firebase config (must match your app config)
firebase.initializeApp({
    apiKey: "AIzaSyAyyoXMZ8UDUd5SMk5g6cGloASpBvLDFBs",
    authDomain: "fireguard-fbe22.firebaseapp.com",
    projectId: "fireguard-fbe22",
    storageBucket: "fireguard-fbe22.firebasestorage.app",
    messagingSenderId: "191728247078",
    appId: "1:191728247078:web:59c9758802de2db9cee1e7",
    measurementId: "G-046VF9C0D8"
});

const messaging = firebase.messaging();

// Handle background messages (when app tab is NOT in focus or browser is closed)
messaging.onBackgroundMessage((payload) => {
    console.log('[firebase-messaging-sw.js] Background message received:', payload);

    const title = payload.notification?.title || '🔥 FIRE ALERT!';
    const body = payload.notification?.body || 'A fire hazard has been detected! Open app immediately!';

    // Define a MASSIVE vibration pattern (simulates a ringer)
    // [vibrate, silent, vibrate, silent, ...]
    const ringerVibration = [
        1000, 500, 1000, 500, 1000, 500, 1000, 500, 1000, 500,
        1000, 500, 1000, 500, 1000, 500, 1000, 500, 1000, 500,
        1000, 500, 1000, 500, 1000, 500, 1000, 500, 1000, 500
    ];

    const notificationOptions = {
        body: body,
        icon: '/static/icons/icon-192x192.png',
        badge: '/static/icons/icon-72x72.png',
        tag: 'fire-alert-permanent', // Same tag so it updates/overwrites if needed
        renotify: true,  // Always vibration/sound even if replacing
        requireInteraction: true, // Notification stays until user taps it (CRITICAL for alert)
        silent: false, // Ensure the notification sound plays
        vibrate: ringerVibration,
        data: {
            url: payload.data?.url || '/adminpanel/',
            timestamp: Date.now()
        },
        actions: [
            { action: 'open_dashboard', title: '🚨 Open Dashboard' },
            { action: 'dismiss', title: 'Dismiss' }
        ]
    };

    // 1. Show the persistent notification (this handles the "beep" and long vibration)
    self.registration.showNotification(title, notificationOptions);

    // 2. Try to inform ANY open window to start the loud alarm IMMEDIATELY
    // Even if the window is in the background, this can trigger the audio buzzer if it was unlocked.
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clientList => {
        for (const client of clientList) {
            client.postMessage({
                type: 'FIRE_ALERT',
                title: title,
                body: body
            });
        }
    });
});

// Handle notification click - opens the site which then plays the buzzer
self.addEventListener('notificationclick', (event) => {
    console.log('[firebase-messaging-sw.js] Notification clicked:', event.action);
    event.notification.close();

    if (event.action === 'dismiss') return;

    // Open the dashboard - this will load the page and the buzzer will play
    const urlToOpen = event.notification.data?.url || '/adminpanel/';

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then((clientList) => {
                // Check if there's already an open window
                for (const client of clientList) {
                    if ('focus' in client) {
                        // Send message to the open page to play alarm
                        client.postMessage({
                            type: 'FIRE_ALERT',
                            title: 'Fire Alert!',
                            body: 'Fire detected! Check dashboard now!'
                        });
                        return client.focus();
                    }
                }
                // No open window - open new one (alarm will play on load)
                if (clients.openWindow) {
                    return clients.openWindow(urlToOpen + '?alarm=true');
                }
            })
    );
});
