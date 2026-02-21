// FireGuard Service Worker - Handles Push Notifications & Offline Caching

const CACHE_NAME = 'fireguard-cache-v2';
const urlsToCache = [
    '/',
    '/static/manifest.json'
];

// Install Event - Cache essential files (best-effort, don't block install)
self.addEventListener('install', (event) => {
    console.log('[ServiceWorker] Installing...');
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[ServiceWorker] Caching app shell');
                // Use individual adds so one failure doesn't block all
                return Promise.allSettled(
                    urlsToCache.map(url => cache.add(url).catch(e => console.log('[SW] Cache skip:', url, e)))
                );
            })
    );
    self.skipWaiting();
});

// Activate Event - Clean old caches
self.addEventListener('activate', (event) => {
    console.log('[ServiceWorker] Activated');
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.filter((name) => name !== CACHE_NAME)
                    .map((name) => caches.delete(name))
            );
        })
    );
    self.clients.claim();
});

// Push Notification Event - This is triggered by FCM
self.addEventListener('push', (event) => {
    console.log('[ServiceWorker] Push Received:', event);

    let data = {
        title: '🔥 Fire Guard Alert!',
        body: 'A potential fire hazard has been detected.',
        icon: '/static/icons/icon-192x192.png',
        badge: '/static/icons/icon-72x72.png',
        tag: 'fire-alert',
        requireInteraction: true, // Keep notification until user interacts
        vibrate: [200, 100, 200, 100, 200], // Vibration pattern
        data: {
            url: '/admin-panel/'
        }
    };

    // If FCM sends custom data, use it
    if (event.data) {
        try {
            const payload = event.data.json();
            data.title = payload.notification?.title || data.title;
            data.body = payload.notification?.body || data.body;
            data.data.url = payload.data?.url || data.data.url;
        } catch (e) {
            console.log('[ServiceWorker] Could not parse push data');
        }
    }

    event.waitUntil(
        self.registration.showNotification(data.title, {
            body: data.body,
            icon: data.icon,
            badge: data.badge,
            tag: data.tag,
            requireInteraction: data.requireInteraction,
            vibrate: data.vibrate,
            data: data.data,
            actions: [
                { action: 'view', title: 'View Dashboard' },
                { action: 'dismiss', title: 'Dismiss' }
            ]
        })
    );
});

// Notification Click Event
self.addEventListener('notificationclick', (event) => {
    console.log('[ServiceWorker] Notification clicked:', event.action);
    event.notification.close();

    if (event.action === 'dismiss') {
        return;
    }

    // Open/focus the dashboard
    const urlToOpen = event.notification.data?.url || '/admin-panel/';
    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then((clientList) => {
                // If dashboard is already open, focus it
                for (const client of clientList) {
                    if (client.url.includes('/admin-panel') && 'focus' in client) {
                        return client.focus();
                    }
                }
                // Otherwise open new window
                if (clients.openWindow) {
                    return clients.openWindow(urlToOpen);
                }
            })
    );
});

// Fetch Event - Network first, fallback to cache
self.addEventListener('fetch', (event) => {
    event.respondWith(
        fetch(event.request)
            .catch(() => caches.match(event.request))
    );
});
