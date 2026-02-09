// preload.js

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Expose a function to open the help/quick start guide.
  openQuickStart: () => ipcRenderer.send('open-quickstart'),
});