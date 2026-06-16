/**
 * api.ts — cliente do backend Flask (/predict).
 *
 * O endereço do backend vem de app.json -> expo.extra.apiHost (não precisa editar código).
 *  - Mesma rede WiFi (Android/Expo Go): IP local da máquina, ex. http://192.168.0.10:5000
 *  - iOS (exige HTTPS): exponha o Flask com ngrok (`ngrok http 5000`) e use a URL https.
 *  - Emulador Android: http://10.0.2.2:5000
 */
import Constants from 'expo-constants';

const extra = (Constants.expoConfig?.extra ?? (Constants.manifest as any)?.extra ?? {}) as {
  apiHost?: string;
};
export const API_HOST = extra.apiHost || 'http://192.168.0.10:5000';

export interface PredictResult {
  classe: string;
  score_experimental: number;
  threshold: number;
  gradcam?: string | null;
  disclaimer: string;
}

export async function predict(imageUri: string): Promise<PredictResult> {
  const form = new FormData();
  const name = imageUri.split('/').pop() || 'image.jpg';
  const ext = (name.split('.').pop() || 'jpg').toLowerCase();
  // React Native FormData aceita { uri, name, type }
  form.append('file', {
    uri: imageUri,
    name,
    type: `image/${ext === 'jpg' ? 'jpeg' : ext}`,
  } as any);

  const resp = await fetch(`${API_HOST}/predict`, {
    method: 'POST',
    body: form,
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  const json = await resp.json();
  if (!resp.ok) throw new Error(json.erro || 'Falha na predição');
  return json as PredictResult;
}
