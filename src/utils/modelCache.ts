class ModelCache {
  private readonly dbName = "piper-tts-cache";
  private readonly storeName = "models";
  private readonly version = 1;
  private db: IDBDatabase | null = null;

  private async init() {
    if (this.db) return this.db;

    return await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(request.result);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: "url" });
          store.createIndex("timestamp", "timestamp", { unique: false });
        }
      };
    });
  }

  async get(url: string) {
    await this.init();

    return await new Promise<ArrayBuffer | null>((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      const request = store.get(url);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const result = request.result as
          | { url: string; data: ArrayBuffer; timestamp: number }
          | undefined;
        if (result) {
          const maxAge = 7 * 24 * 60 * 60 * 1000;
          if (Date.now() - result.timestamp < maxAge) {
            resolve(result.data);
            return;
          } else {
            void this.delete(url);
          }
        }
        resolve(null);
      };
    });
  }

  async set(url: string, data: ArrayBuffer) {
    await this.init();

    return await new Promise<void>((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);
      const request = store.put({
        url,
        data,
        timestamp: Date.now(),
      });

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }

  async delete(url: string) {
    await this.init();

    return await new Promise<void>((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);
      const request = store.delete(url);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }
}

export async function cachedFetch(url: string) {
  const cache = new ModelCache();
  const cachedData = await cache.get(url);
  if (cachedData) {
    return new Response(cachedData);
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.arrayBuffer();
  await cache.set(url, data);

  return new Response(data, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}
