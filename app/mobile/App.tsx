/**
 * CardioIA Fase 4 — App mobile (Ir Além 2).
 * Expo / React Native. Tela única: upload/captura → resultado da classificação.
 * Consome o MESMO backend Flask (/predict). Configure o host em src/services/api.ts.
 *
 * Design: "Clinical Precision Narrative" — Deep Medical Blue, tipografia mono para o
 * score experimental, disclaimer não dispensável, régua de decisão (sem barra de saúde).
 * Governança: exibe disclaimer; trata o valor como "score experimental", não diagnóstico.
 * Interface 100% em pt-br.
 *
 * Setup:
 *   npm install
 *   npx expo start         // abrir no Expo Go (dispositivo físico p/ câmera)
 */
import React, { useState } from 'react';
import {
  ActivityIndicator, Image, Platform, SafeAreaView, ScrollView,
  StatusBar, StyleSheet, Switch, Text, TouchableOpacity, View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { predict, PredictResult } from './src/services/api';

// ---- Paleta Clinical Precision Narrative ----
const C = {
  surface: '#f7f9fb',
  card: '#ffffff',
  containerLow: '#f2f4f6',
  containerHigh: '#e6e8ea',
  onSurface: '#191c1e',
  onSurfaceVariant: '#424751',
  outline: '#737783',
  outlineVariant: '#c2c6d3',
  primary: '#00346f',
  primaryContainer: '#004a99',
  onPrimary: '#ffffff',
  clinicalBlueDark: '#003366',
  clinicalBlueLight: '#e0f2fe',
  tertiary: '#71001d',
  disclaimerBg: '#fff1f2',
  errorContainer: '#ffdad6',
  onErrorContainer: '#93000a',
};

const MONO = Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' });

const LABELS_PT: Record<string, string> = {
  Cardiomegaly: 'Cardiomegalia',
  'No Finding': 'Sem achados',
};

const DISCLAIMER =
  'Isto NÃO é um dispositivo médico. Protótipo acadêmico (FIAP), sem validação clínica e ' +
  'que NÃO deve ser usado para diagnóstico real. O valor é um score experimental, não ' +
  'probabilidade clínica. Não envie exames reais identificáveis.';

export default function App() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showCam, setShowCam] = useState(false);

  async function pickImage(fromCamera: boolean) {
    setError(null);
    setResult(null);
    setShowCam(false);
    const perm = fromCamera
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      setError('Permissão negada.');
      return;
    }

    const res = fromCamera
      ? await ImagePicker.launchCameraAsync({ quality: 0.8 })
      : await ImagePicker.launchImageLibraryAsync({ quality: 0.8 });
    if (res.canceled) return;

    const uri = res.assets[0].uri;
    setImageUri(uri);
    await analyze(uri);
  }

  async function analyze(uri: string) {
    setLoading(true);
    setError(null);
    try {
      const r = await predict(uri);
      setResult(r);
    } catch (e: any) {
      setError(e?.message ?? 'Falha ao analisar.');
    } finally {
      setLoading(false);
    }
  }

  const isFlag = result?.classe === 'Cardiomegaly';
  const labelPt = result ? LABELS_PT[result.classe] ?? result.classe : '';
  const scorePos = result ? Math.min(1, Math.max(0, Number(result.score_experimental))) : 0;
  const thrPos = result ? Number(result.threshold) : 0.5;

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" backgroundColor={C.primary} />

      {/* App bar */}
      <View style={styles.appbar}>
        <View style={styles.mark}>
          <Text style={styles.markGlyph}>♥</Text>
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.appbarTitle}>CardioIA</Text>
          <Text style={styles.appbarSub}>Triagem de cardiomegalia · Visão Computacional</Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={styles.container}>
        {/* Disclaimer não dispensável */}
        <View style={styles.disclaimer}>
          <View style={styles.disclaimerBar} />
          <Text style={styles.disclaimerText}>{DISCLAIMER}</Text>
        </View>

        {/* Ações */}
        <Text style={styles.sectionLabel}>ENVIAR EXAME</Text>
        <View style={styles.row}>
          <TouchableOpacity style={styles.btnPrimary} onPress={() => pickImage(false)} activeOpacity={0.85}>
            <Text style={styles.btnPrimaryText}>Galeria</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.btnSecondary} onPress={() => pickImage(true)} activeOpacity={0.85}>
            <Text style={styles.btnSecondaryText}>Câmera</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.hint}>A imagem é redimensionada para 224×224 px — a entrada da rede neural.</Text>

        {loading && (
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color={C.primary} />
            <Text style={styles.loadingText}>Analisando…</Text>
          </View>
        )}

        {error && (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {result && imageUri && (
          <>
            {/* Imagem + auditoria Grad-CAM */}
            <View style={styles.card}>
              <Text style={styles.sectionLabel}>ENTRADA DA CNN (224×224)</Text>
              <View style={styles.xrayFrame}>
                <Image
                  source={{ uri: showCam && result.gradcam ? result.gradcam : imageUri }}
                  style={styles.xray}
                  resizeMode="contain"
                  accessibilityLabel={
                    showCam
                      ? 'Mapa de calor Grad-CAM sobreposto ao raio-X'
                      : 'Raio-X de tórax enviado'
                  }
                />
                <View style={styles.xrayMeta}>
                  <Text style={styles.xrayMetaText}>{showCam ? 'Grad-CAM (auditoria)' : 'raio-X original'}</Text>
                  <Text style={styles.xrayMetaText}>224×224</Text>
                </View>
              </View>

              <View style={styles.toggleRow}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.toggleTitle}>Visualização de auditoria</Text>
                  <Text style={styles.toggleSub}>Grad-CAM · onde o modelo olhou</Text>
                </View>
                <Switch
                  value={showCam}
                  onValueChange={setShowCam}
                  disabled={!result.gradcam}
                  trackColor={{ false: C.containerHigh, true: C.primary }}
                  thumbColor="#ffffff"
                />
              </View>
              {!result.gradcam && <Text style={styles.toggleSub}>Grad-CAM indisponível para esta imagem.</Text>}
            </View>

            {/* Score card */}
            <View style={styles.card}>
              <Text style={styles.scoreLabel}>Score experimental</Text>
              <Text style={styles.scoreValue}>{Number(result.score_experimental).toFixed(4)}</Text>
              <Text style={styles.scoreSub}>
                Limiar de decisão: <Text style={styles.mono}>{thrPos.toFixed(2)}</Text>
              </Text>

              {/* Régua de decisão (não é barra de saúde) */}
              <View style={styles.rulerTrack}>
                <View style={[styles.rulerThr, { left: `${thrPos * 100}%` }]} />
                <View style={[styles.rulerDot, { left: `${scorePos * 100}%` }]} />
              </View>
              <View style={styles.rulerScale}>
                <Text style={styles.rulerScaleText}>0.0</Text>
                <Text style={styles.rulerScaleText}>1.0</Text>
              </View>

              <View style={styles.verdict}>
                <Text style={styles.sectionLabel}>CLASSE SUGERIDA PELO MODELO</Text>
                <View style={[styles.classChip, isFlag ? styles.classFlag : styles.classClear]}>
                  <View style={[styles.classMarker, { backgroundColor: isFlag ? C.primary : C.outline }]} />
                  <Text style={[styles.classText, { color: isFlag ? C.clinicalBlueDark : C.onSurfaceVariant }]}>
                    {labelPt}
                  </Text>
                </View>
                <Text style={styles.verdictNote}>
                  {isFlag
                    ? 'O modelo sinalizou padrões compatíveis com cardiomegalia. Achado experimental que exige confirmação por profissional de saúde.'
                    : 'O modelo não sinalizou cardiomegalia neste exame. A ausência de sinalização não substitui avaliação clínica.'}
                </Text>
              </View>
            </View>
          </>
        )}

        <Text style={styles.footer}>
          CardioIA · FIAP — Fase 4 · Retenção zero: a imagem é processada em memória e não é armazenada.
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.surface },
  appbar: {
    backgroundColor: C.primary,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  mark: {
    width: 38, height: 38, borderRadius: 6,
    backgroundColor: 'rgba(255,255,255,0.10)',
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.18)',
    alignItems: 'center', justifyContent: 'center',
  },
  markGlyph: { color: C.clinicalBlueLight, fontSize: 18 },
  appbarTitle: { color: C.onPrimary, fontSize: 18, fontWeight: '700' },
  appbarSub: { color: C.clinicalBlueLight, fontSize: 12 },

  container: { padding: 16, paddingBottom: 40 },

  disclaimer: {
    flexDirection: 'row',
    backgroundColor: C.disclaimerBg,
    borderColor: C.errorContainer, borderWidth: 1,
    borderRadius: 6,
    overflow: 'hidden',
    marginBottom: 20,
  },
  disclaimerBar: { width: 6, backgroundColor: C.tertiary },
  disclaimerText: { flex: 1, color: '#5b0016', fontSize: 13, fontWeight: '600', lineHeight: 18, padding: 12 },

  sectionLabel: {
    fontSize: 12, fontWeight: '700', letterSpacing: 0.6,
    color: C.onSurfaceVariant, marginBottom: 8,
  },
  row: { flexDirection: 'row', gap: 12 },
  btnPrimary: {
    flex: 1, backgroundColor: C.primary, paddingVertical: 14, borderRadius: 4, alignItems: 'center',
  },
  btnPrimaryText: { color: C.onPrimary, fontWeight: '600', fontSize: 15 },
  btnSecondary: {
    flex: 1, backgroundColor: 'transparent', paddingVertical: 14, borderRadius: 4, alignItems: 'center',
    borderWidth: 1, borderColor: C.outlineVariant,
  },
  btnSecondaryText: { color: C.primaryContainer, fontWeight: '600', fontSize: 15 },
  hint: { color: C.onSurfaceVariant, fontSize: 13, marginTop: 10, lineHeight: 18 },

  loadingBox: { alignItems: 'center', marginTop: 24 },
  loadingText: { color: C.onSurfaceVariant, marginTop: 8, fontSize: 13 },

  errorBox: {
    backgroundColor: C.errorContainer, borderColor: '#f5b3ad', borderWidth: 1,
    borderRadius: 4, padding: 12, marginTop: 16,
  },
  errorText: { color: C.onErrorContainer, fontWeight: '500' },

  card: {
    backgroundColor: C.card, borderRadius: 4, padding: 16, marginTop: 16,
    borderWidth: 1, borderColor: C.outlineVariant,
  },

  xrayFrame: {
    width: 224, height: 224, alignSelf: 'center',
    backgroundColor: '#000', borderWidth: 1, borderColor: C.outline,
    borderRadius: 0, overflow: 'hidden',
  },
  xray: { width: '100%', height: '100%' },
  xrayMeta: {
    position: 'absolute', left: 0, right: 0, bottom: 0,
    backgroundColor: 'rgba(0,30,64,0.78)', flexDirection: 'row', justifyContent: 'space-between',
    paddingHorizontal: 7, paddingVertical: 4,
  },
  xrayMetaText: { color: '#dbe7ff', fontSize: 10, fontWeight: '700', letterSpacing: 0.4, textTransform: 'uppercase' },

  toggleRow: {
    flexDirection: 'row', alignItems: 'center', gap: 12, marginTop: 14,
    backgroundColor: C.containerLow, borderColor: C.outlineVariant, borderWidth: 1,
    borderRadius: 6, padding: 12,
  },
  toggleTitle: { fontSize: 13, fontWeight: '600', color: C.onSurface },
  toggleSub: { fontSize: 11.5, color: C.onSurfaceVariant, marginTop: 2 },

  scoreLabel: { fontSize: 16, fontWeight: '600', color: C.onSurface, marginBottom: 8 },
  scoreValue: { fontFamily: MONO, fontSize: 44, fontWeight: '500', color: C.clinicalBlueDark },
  scoreSub: { fontSize: 13, color: C.onSurfaceVariant, marginTop: 6 },
  mono: { fontFamily: MONO },

  rulerTrack: {
    height: 6, backgroundColor: C.containerHigh, borderRadius: 999,
    marginTop: 20, marginBottom: 6, position: 'relative',
  },
  rulerThr: { position: 'absolute', top: -7, bottom: -7, width: 2, backgroundColor: C.outline, marginLeft: -1 },
  rulerDot: {
    position: 'absolute', top: -4, width: 14, height: 14, borderRadius: 7,
    backgroundColor: C.clinicalBlueDark, borderWidth: 2, borderColor: '#fff', marginLeft: -7,
  },
  rulerScale: { flexDirection: 'row', justifyContent: 'space-between' },
  rulerScaleText: { fontFamily: MONO, fontSize: 11, color: C.outline },

  verdict: { marginTop: 18, paddingTop: 16, borderTopWidth: 1, borderTopColor: C.outlineVariant },
  classChip: {
    flexDirection: 'row', alignItems: 'center', gap: 9, alignSelf: 'flex-start',
    paddingVertical: 9, paddingHorizontal: 14, borderRadius: 8, borderWidth: 1,
  },
  classFlag: { backgroundColor: C.clinicalBlueLight, borderColor: '#b6d8f5' },
  classClear: { backgroundColor: C.containerHigh, borderColor: C.outlineVariant },
  classMarker: { width: 9, height: 9, borderRadius: 5 },
  classText: { fontWeight: '700', fontSize: 15 },
  verdictNote: { fontSize: 13, color: C.onSurfaceVariant, lineHeight: 19, marginTop: 10 },

  footer: { fontSize: 12, color: C.onSurfaceVariant, lineHeight: 17, marginTop: 24, textAlign: 'center' },
});
