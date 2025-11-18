import React, { useState, useEffect, useCallback, useRef } from 'react';
import { View, Text, TextInput, StyleSheet, Alert, Pressable as RNPressable, ScrollView, AccessibilityInfo } from 'react-native';
import * as Speech from 'expo-speech';
import * as Haptics from 'expo-haptics';
import { Audio } from 'expo-av';
import axios from 'axios';

const questions = [
  {
    key: 'crowdTolerance',
    text: 'First question. How sensitive are you to crowded spaces? Please choose one of three levels: High, meaning you prefer quiet spaces. Medium, or Low, meaning crowds are fine.'
  },
  {
    key: 'seatingPreference',
    text: 'When you board a bus, which area do you prefer to sit in? The front, the middle, or the rear?'
  },
  {
    key: 'conditionTriggers',
    text: 'Are you concerned about the general condition of the bus? For example, problems like broken seats, visible trash, or loud engine noise.'
  },
  {
    key: 'detailLevel',
    text: 'How detailed should my descriptions of the route and surroundings be? Choose either Short, for just the facts, or Detailed, to include safety and other context.'
  },
];

const IP = '192.168.50.131'; // IMPORTANT: Verify this IP matches your local server
const GOOGLE_API_KEY = 'AIzaSyCFOLP0EfP2OgqU6NHLjjUpktvO4fE8xGE';

export default function OnboardingScreen({ navigation, route }) {
  const userId = route.params?.userId ?? 'test-user';
  const [idx, setIdx] = useState(0);
  const [answer, setAnswer] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [recording, setRecording] = useState(null);
  const [isAwaitingConfirmation, setIsAwaitingConfirmation] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [shouldSpeakNext, setShouldSpeakNext] = useState(true);



  // --- Confirmation Handlers ---
  const saveAndNext = async () => {
    if (isRecording) { setIsRecording(false); }
    const q = questions[idx];

    if (!answer.trim()) {
      Alert.alert('Answer Required', 'Please provide an answer or say "skip" to continue.');
      return;
    }

    Speech.stop();

    try {
      await axios.post(`http://${IP}:8000/store-preference`, {
        userId,
        key: q.key,
        value: (answer || 'skipped').toLowerCase().trim(),
      }, { headers: { 'Content-Type': 'application/json' } });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (e) {
      Alert.alert('Save Failed', `Could not reach backend: ${e.message}. Check if server is running on ${IP}:8000.`);
      return;
    }

    setIsSpeaking(false);

    if (idx < questions.length - 1) {
      setAnswer('');
      setIsAwaitingConfirmation(false);
      setIdx(idx + 1);
      setShouldSpeakNext(true);
    } else {
      navigation.replace('Landing', { userId });
      Speech.speak('Welcome to VisionaryGuide! Setup complete.', {
        onDone: () => { setIsSpeaking(false); }
      });
    }
  };


  const confirmAnswer = useCallback(() => {
    Speech.stop();
    setIsSpeaking(false);
    setIsAwaitingConfirmation(false);
    setShouldSpeakNext(false);
    saveAndNext();
  }, [idx, answer]);

  const cancelAndReanswer = useCallback(() => {
    Speech.stop();
    setIsSpeaking(false);

    setAnswer('');
    setIsAwaitingConfirmation(false);

    const speechText = `Please re-answer the question: ${questions[idx].text}`;
    setIsSpeaking(true);
    Speech.speak(speechText, {
      onDone: () => setIsSpeaking(false)
    });
    AccessibilityInfo.announceForAccessibility("Answer cancelled. Please record again.");
    setShouldSpeakNext(true);
  }, [idx]);


  // --- Voice Command/Answer Recording Logic ---

  // Helper function to handle transcription
  const processTranscription = useCallback(async (uri, isCommandMode = false) => {
    // Ensure we are not recording anymore
    setIsRecording(false);

    try {
      const response = await fetch(uri);
      const blob = await response.blob();
      const reader = new FileReader();

      reader.onloadend = async () => {
        const base64 = reader.result.split(',')[1];
        const res = await fetch(
          `https://speech.googleapis.com/v1/speech:recognize?key=${GOOGLE_API_KEY}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              config: { encoding: 'WEBM_OPUS', sampleRateHertz: 48000, languageCode: 'en-US' },
              audio: { content: base64 },
            }),
          }
        );
        const data = await res.json();
        const text = (data.results?.[0]?.alternatives?.[0]?.transcript || '').trim().toLowerCase();

        if (isCommandMode) {
          // 🛑 COMMAND MODE DISABLED: User must use the buttons.
          setIsSpeaking(true);
          Speech.speak("Please use the 'Proceed' or 'Re-answer' buttons below to continue.", {
            onDone: () => setIsSpeaking(false)
          });
          return;
        } else {
          // ANSWER MODE LOGIC 
          if (!text) {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
            Speech.stop();
            setIsSpeaking(true);
            Speech.speak("I heard nothing. Please hold the screen and speak your answer clearly when you hear the chime.", {
              onDone: () => setIsSpeaking(false)
            });
            return;
          }

          setAnswer(text);
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

          setIsAwaitingConfirmation(true);
          setIsSpeaking(true);

          Speech.speak(`I heard, "${text}". Do you want to save this answer?`, {
            onDone: () => {
              setIsSpeaking(false);
            }
          });
        }
      };
      reader.readAsDataURL(blob);
    } catch (e) {
      Alert.alert('Speech Recognition Error', 'Using fallback response');
      if (!isCommandMode) {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
        Speech.stop();
        setIsSpeaking(true);
        Speech.speak("I could not connect to the speech service. Please try recording again, or use the text box.", {
          onDone: () => setIsSpeaking(false)
        });
        setIsAwaitingConfirmation(false);
      } else {
        setIsSpeaking(true);
        Speech.speak("Voice input failed. Please use the buttons below.", {
          onDone: () => setIsSpeaking(false)
        });
      }
    }
  }, [confirmAnswer, cancelAndReanswer]);


  // Starts the recording session 
  const setupRecording = async () => {
    try {
      // 1. AGGRESSIVE CLEANUP: Ensure any lingering object is destroyed before creation.
      if (recording) {
        console.log("Unloading lingering object before setup.");
        await recording.stopAndUnloadAsync();
        setRecording(null);
        setIsRecording(false);
        // Force a small pause to allow OS to release the resource
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // 2. GET PERMISSION
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Microphone Access Required', 'Please grant microphone access.');
        return null;
      }

      // 3. FORCE AUDIO MODE RESET (Active recording mode - USING NUMERICAL CONSTANTS)
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        // 0 = DO_NOT_MIX (iOS default we were trying to set)
        interruptionModeIOS: 0,
        // 2 = DO_NOT_MIX (Android constant fix)
        interruptionModeAndroid: 2,
      });

      // 4. CREATE NEW RECORDING OBJECT
      const { recording: newRecording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      setRecording(newRecording);
      return newRecording;
    } catch (err) {
      console.error('Failed to start recording:', err);
      // If creation fails, ensure states are false
      setRecording(null);
      setIsRecording(false);
      return null;
    }
  };

  // Stops the recording session
  const stopAndCleanupRecording = async (currentRecording, isCommandMode) => {
    if (!currentRecording) return null;

    // Step 1: Stop and unload the object (releases URI)
    await currentRecording.stopAndUnloadAsync();
    const uri = currentRecording.getURI();

    // Step 2: Immediately clear the recording object state 
    setRecording(null);
    setIsRecording(false);

    // CRITICAL STEP: Reset audio mode (USING NUMERICAL CONSTANTS)
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: false, // Turn off recording mode
      playsInSilentModeIOS: true,
      shouldDuckAndroid: true,
      // 0 = DO_NOT_MIX (iOS)
      interruptionModeIOS: 0,
      // 2 = DO_NOT_MIX (Android)
      interruptionModeAndroid: 2,
      playThroughEarpieceAndroid: false,
      staysActiveInBackground: false,
    });


    return uri;
  }


  // Manual start recording (used for question answering)
  const startRecording = async () => {
    // Explicitly prevent recording if on confirmation screen or system is speaking
    if (isSpeaking || isAwaitingConfirmation) return;

    const rec = await setupRecording();
    if (rec) {
      setIsRecording(true); // ⭐ Only set state if setup was successful
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      AccessibilityInfo.announceForAccessibility('Recording started');
    }
  };

  // Manual stop recording (used for question answering)
  const stopRecording = async () => {
    if (!recording) return;

    // Stop the recording and get the URI
    const uri = await stopAndCleanupRecording(recording, isAwaitingConfirmation);

    // CRITICAL: Small delay to ensure native resource is fully released before transcription starts
    await new Promise(resolve => setTimeout(resolve, 50));

    if (uri) {
      processTranscription(uri, isAwaitingConfirmation);
    }
  };


  // --- Core Speech Logic: Announce New Question ---
  useEffect(() => {
    if (!isAwaitingConfirmation && shouldSpeakNext) {
      Speech.stop();
      setIsSpeaking(true);
      Speech.speak(questions[idx].text, {
        rate: 0.9,
        onDone: () => setIsSpeaking(false)
      });
      AccessibilityInfo.announceForAccessibility(
        `Question ${idx + 1} of ${questions.length}. ${questions[idx].text}`
      );
      setShouldSpeakNext(true);
    }
  }, [idx, isAwaitingConfirmation, shouldSpeakNext]);

  // Clean up only on component unmount
  useEffect(() => {
    return () => {
      Speech.stop();
      // If a recording object is still active, make sure to unload it.
      if (recording) {
        recording.stopAndUnloadAsync().catch(e => console.log("Failed to unload on unmount:", e));
      }
    };
  }, [recording]);


  const progress = ((idx + 1) / questions.length) * 100;

  // ----------------------------------------------------
  // RENDER 
  // ----------------------------------------------------
  if (isAwaitingConfirmation) {
    return (
      <View // ⬅️ Using a View ensures no accidental touch-to-record events are possible
        style={[styles.scrollContainer, styles.confirmationContainer]}
        accessible={true}
        accessibilityLabel="Confirmation screen. Your recorded answer is shown."
      >
        <Text style={styles.confirmationTitle}>Confirm Your Answer</Text>
        <View style={styles.queryBox}>
          <Text style={styles.queryText}>"{answer}"</Text>
        </View>
        <Text style={styles.confirmationPrompt}>
          {isSpeaking ? "System speaking instructions..." : "Please use the buttons below to proceed or re-answer."}
        </Text>

        <View style={styles.buttonContainer}>
          {/* Buttons are disabled only when the system is speaking or recording */}
          <RNPressable style={[styles.confirmButton, styles.proceedButton]} onPress={confirmAnswer} disabled={isSpeaking || isRecording}>
            <Text style={styles.buttonText}>Proceed</Text>
          </RNPressable>
          <RNPressable style={[styles.confirmButton, styles.cancelButton]} onPress={cancelAndReanswer} disabled={isSpeaking || isRecording}>
            <Text style={styles.buttonText}>Re-answer</Text>
          </RNPressable>
        </View>
      </View>
    );
  }

  return (
    <RNPressable
      style={{ flex: 1 }}
      accessible={true}
      accessibilityLabel="Onboarding questionnaire"
      onLongPress={startRecording}
      onPressOut={stopRecording}
      // Disable touch events when recording OR speaking is active
      pointerEvents={isRecording || isSpeaking ? 'none' : 'auto'}
    >
      <ScrollView
        contentContainerStyle={styles.scrollContainer}
        scrollEventThrottle={16}
      >
        <View style={styles.container}>
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.logo} accessible={true} accessibilityRole="header" accessibilityLabel="Visionary Guide Setup">Visionary Guide</Text>
            <Text style={styles.subtitle} accessible={true} accessibilityLabel={`Setting up your preferences. Question ${idx + 1} of ${questions.length}`}>Let's personalize your experience</Text>
          </View>

          {/* Progress Bar */}
          <View style={styles.progressContainer} accessible={true} accessibilityRole="progressbar" accessibilityLabel={`Progress: ${Math.round(progress)}% complete`} accessibilityValue={{ now: progress, min: 0, max: 100 }}>
            <View style={styles.progressBar}>
              <View style={[styles.progressFill, { width: `${progress}%` }]} />
            </View>
            <Text style={styles.progressText}>{idx + 1} / {questions.length}</Text>
          </View>

          {/* Question Card */}
          <View style={styles.questionCard}>
            <Text style={styles.questionLabel} accessible={true} accessibilityRole="text">Question {idx + 1}</Text>
            <Text style={styles.questionText} accessible={true} accessibilityRole="text" accessibilityLabel={questions[idx].text}>{questions[idx].text}</Text>
          </View>

          {/* Answer Input */}
          <View style={styles.inputContainer}>
            <Text style={styles.inputLabel} accessible={true}>Your Answer</Text>
            <TextInput style={styles.input} value={answer} onChangeText={setAnswer} placeholder="Type your answer here..." placeholderTextColor="#999" accessible={true} accessibilityLabel="Answer input field" accessibilityHint="Enter your answer or use voice input below" multiline numberOfLines={3} editable={!isRecording && !isSpeaking} />
          </View>

          {/* Voice Input Indicator */}
          <View
            style={[
              styles.micButton,
              isRecording && styles.micRecording,
              isSpeaking && { opacity: 0.5, backgroundColor: '#666' }
            ]}
            accessible={true}
            accessibilityRole="text"
            accessibilityLabel={isRecording ? 'Recording in progress' : isSpeaking ? 'System is speaking, recording is paused' : 'Press and hold anywhere on screen to record'}
          >
            <View style={styles.micButtonContent}>
              <View style={styles.micIcon}>
                <Text style={styles.micIconText}>🎤</Text>
              </View>
              <Text style={styles.micText}>
                {isRecording ? 'Recording...' : isSpeaking ? 'System Speaking...' : 'Hold Anywhere to Speak'}
              </Text>
            </View>
          </View>

          {/* Navigation Buttons */}
          <View style={styles.buttonRow}>
            <RNPressable
              style={({ pressed }) => [
                styles.nextButton,
                pressed && styles.nextPressed
              ]}
              onPress={saveAndNext}
              disabled={isRecording || isSpeaking || !answer.trim()}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel={idx < questions.length - 1 ? 'Next question' : 'Complete setup'}
              accessibilityHint="Saves your answer and continues"
            >
              <Text style={styles.nextText}>
                {idx < questions.length - 1 ? 'Next' : 'Complete'}
              </Text>
            </RNPressable>
          </View>

          {/* Helper Text */}
          <Text
            style={styles.helperText}
            accessible={true}
            accessibilityRole="text"
          >
            💡 Tip: Hold anywhere on the screen to speak your answer
          </Text>
        </View>
      </ScrollView>
    </RNPressable>
  );
}

const styles = StyleSheet.create({
  scrollContainer: { flexGrow: 1, backgroundColor: '#F8F9FA', },
  container: { flex: 1, padding: 20, paddingTop: 60, },
  header: { marginBottom: 20, alignItems: 'center', },
  logo: { fontSize: 32, fontWeight: '800', color: '#1A1A1A', marginBottom: 8, letterSpacing: 0.5, },
  subtitle: { fontSize: 16, color: '#666', textAlign: 'center', },
  progressContainer: { marginBottom: 8, },
  progressBar: { height: 8, backgroundColor: '#E0E0E0', borderRadius: 4, overflow: 'hidden', marginBottom: 2, },
  progressFill: { height: '100%', backgroundColor: '#007AFF', borderRadius: 4, },
  progressText: { fontSize: 14, color: '#666', textAlign: 'right', fontWeight: '600', },
  questionCard: { backgroundColor: '#FFFFFF', borderRadius: 16, padding: 24, marginBottom: 24, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.08, shadowRadius: 8, elevation: 3, },
  questionLabel: { fontSize: 16, color: '#007AFF', fontWeight: '700', marginBottom: 2, letterSpacing: 0.5, textTransform: 'uppercase', },
  questionText: { fontSize: 14, color: '#1A1A1A', lineHeight: 28, fontWeight: '600', },
  inputContainer: { marginBottom: 10, },
  inputLabel: { fontSize: 14, color: '#666', fontWeight: '600', marginBottom: 1, },
  input: { backgroundColor: '#FFFFFF', borderWidth: 2, borderColor: '#E0E0E0', padding: 16, borderRadius: 12, fontSize: 16, minHeight: 80, textAlignVertical: 'top', color: '#1A1A1A', },
  micButton: { backgroundColor: '#FF9500', borderRadius: 16, marginBottom: 16, shadowColor: '#FF9500', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 8, elevation: 6, },
  micPressed: { backgroundColor: '#CC7700', transform: [{ scale: 0.97 }], },
  micRecording: { backgroundColor: '#FF3B30', shadowColor: '#FF3B30', },
  micButtonContent: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: 18, },
  micIcon: { width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center', marginRight: 12, },
  micIconText: { fontSize: 20, },
  micText: { color: '#FFFFFF', fontWeight: '700', fontSize: 18, },
  buttonRow: { flexDirection: 'row', marginBottom: 16, },
  nextButton: { flex: 1, backgroundColor: '#34C759', padding: 18, borderRadius: 16, alignItems: 'center', shadowColor: '#34C759', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 8, elevation: 6, },
  nextPressed: { backgroundColor: '#2a9d4f', transform: [{ scale: 0.97 }], },
  nextText: { color: '#FFFFFF', fontWeight: '700', fontSize: 18, },
  helperText: { fontSize: 11, color: '#666', textAlign: 'center', },
  confirmationContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 40, backgroundColor: '#1C1C1E', },
  confirmationTitle: { fontSize: 26, fontWeight: '900', color: 'white', marginBottom: 30, textAlign: 'center', },
  queryBox: { backgroundColor: 'rgba(255, 255, 255, 0.1)', padding: 25, borderRadius: 15, width: '100%', alignItems: 'center', marginBottom: 30, },
  queryText: { fontSize: 22, color: '#007AFF', fontWeight: '600', textAlign: 'center', fontStyle: 'italic', },
  confirmationPrompt: { fontSize: 18, color: '#A0A0A0', marginBottom: 50, fontWeight: '500', textAlign: 'center', },
  buttonContainer: { flexDirection: 'row', width: '100%', justifyContent: 'space-between', },
  confirmButton: { paddingVertical: 15, paddingHorizontal: 20, borderRadius: 12, width: '48%', alignItems: 'center', justifyContent: 'center', },
  proceedButton: { backgroundColor: '#34C759', },
  cancelButton: { backgroundColor: '#FF3B30', },
  buttonText: { color: 'white', fontSize: 18, fontWeight: '700', },
});
