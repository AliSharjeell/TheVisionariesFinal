// screens/LandingScreen.jsx (FIXED)
import React, { useEffect, useState, useRef } from 'react';
import { View, Text, StyleSheet, Pressable, Dimensions, AccessibilityInfo, Animated, Alert, ActivityIndicator } from 'react-native';
import * as Speech from 'expo-speech';
import * as Haptics from 'expo-haptics';
import { Audio } from 'expo-av';
import { CameraView, useCameraPermissions, Camera } from 'expo-camera';
import { LinearGradient } from 'expo-linear-gradient';
import { Feather } from '@expo/vector-icons';
import axios from 'axios'; // 👈 NEW
import { Platform } from 'react-native';

const PHRASE = 'hello';
const IP = '192.168.50.131'; // 👈 NEW: Define your backend IP
const GOOGLE_API_KEY = 'AIzaSyCFOLP0EfP2OgqU6NHLjjUpktvO4fE8xGE';
const { width, height } = Dimensions.get('window');

// Component to handle the colorful, pulsing glow effect
const AnimatedGlow = ({ isRecording, pulseAnim }) => {
  // Interpolate the scale for both gentle (idle) and strong (listening) pulse
  const gentleScale = pulseAnim.interpolate({
    inputRange: [1, 1.15],
    outputRange: [1, 1.05],
  });
  const strongScale = pulseAnim.interpolate({
    inputRange: [1, 1.15],
    outputRange: [1.1, 1.25],
  });

  return (
    <Animated.View
      style={[
        styles.glowWrapper,
        {
          // Scale based on state
          transform: [{ scale: isRecording ? strongScale : gentleScale }],
          // Opacity based on state
          opacity: isRecording ? 0.8 : 0.6,
        },
      ]}
    >
      <LinearGradient
        colors={['rgba(66, 133, 244, 0.4)', 'rgba(52, 168, 83, 0.4)', 'rgba(251, 188, 5, 0.4)', 'rgba(234, 67, 53, 0.4)']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.glowGradient}
      />
    </Animated.View>
  );
};

export default function LandingScreen({ route }) {
  const userId = route.params?.userId ?? 'test-user';
  const [isRecording, setIsRecording] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [recording, setRecording] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const cameraRef = useRef(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [userPreferences, setUserPreferences] = useState({}); // 👈 NEW: To store the preferences
  const [isPrefsLoading, setIsPrefsLoading] = useState(true); // 👈 NEW: Loading indicator for prefs
  const [confirmingQuery, setConfirmingQuery] = useState(null); // Holds the transcribed text
  const [isAwaitingConfirmation, setIsAwaitingConfirmation] = useState(false); // Controls the new confirmation UI

  const pulseAnim = useRef(new Animated.Value(1)).current;

  const fetchUserPreferences = async () => {
    setIsPrefsLoading(true);
    try {
      const response = await axios.get(`http://${IP}:8000/get-prefs/${userId}`);
      setUserPreferences(response.data);
      console.log("Preferences loaded:", response.data);
    } catch (error) {
      console.error("Failed to load user preferences:", error.message);
      // Optionally, prompt user to go back to onboarding
    } finally {
      setIsPrefsLoading(false);
    }
  };

  useEffect(() => {
    Camera.requestCameraPermissionsAsync();
    fetchUserPreferences();
    AccessibilityInfo.announceForAccessibility('VisionaryGuide ready. Hold the microphone to speak.');
  }, []);

  useEffect(() => {
    pulseAnim.stopAnimation();

    if (isRecording || isProcessing) {
      // Strong pulse for listening/processing state
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.15, duration: 750, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 750, useNativeDriver: true }),
        ])
      ).start();
    } else {
      // Gentle pulse for idle state
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.05, duration: 1500, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 1500, useNativeDriver: true }),
        ])
      ).start();
    }

    return () => pulseAnim.stopAnimation();
  }, [isRecording, isProcessing, pulseAnim]);

  

  const startRecording = async () => {
    if (isRecording || isProcessing) return;

    try {
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Mic Access', 'Microphone permission denied.');
        return;
      }

      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
      const { recording } = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      setRecording(recording);
      setIsRecording(true);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

      // Reduced speech volume/duration to make it less intrusive when holding the button
      // Speech.speak('Recording...');
      AccessibilityInfo.announceForAccessibility('Recording started.');

    } catch (err) {
      console.error("Recording error:", err);
      Alert.alert('Error', 'Recording failed.');
    }
  };


  const sendRequestToAI = async (imageFile, userPrompt, userPreference) => {
  if (!imageFile) {
    throw new Error("No image file provided!");
  }

  try {
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("user_prompt", userPrompt);
    formData.append("userPreferences", JSON.stringify(userPreference));

    const response = await fetch(`http://${IP}:8000/askAi/`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json(); // parse JSON response
    return data.response; // extract actual AI response
  } catch (error) {
    console.error("Error in sendRequestToAI:", error);
    throw error;
  }
};




 const stopRecording = async () => {
  if (!recording) return;

  setIsRecording(false);
  setIsProcessing(true);

  try {
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();

    // Audio → Google STT
    const audioResponse = await fetch(uri);
    const audioBlob = await audioResponse.blob();
    const reader = new FileReader();

    reader.onloadend = async () => {
      const base64Audio = reader.result.split(',')[1];

      const sttRes = await fetch(
        `https://speech.googleapis.com/v1/speech:recognize?key=${GOOGLE_API_KEY}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            config: {
              encoding: 'WEBM_OPUS',
              sampleRateHertz: 48000,
              languageCode: 'en-US',
            },
            audio: { content: base64Audio },
          }),
        }
      );

      const sttData = await sttRes.json();
      const text = (sttData.results?.[0]?.alternatives?.[0]?.transcript || '').toLowerCase();
      const query = text.replace(PHRASE, '').trim() || 'what do you see';
      setConfirmingQuery(query);

      if (cameraRef.current) {
        try {
          const photo = await cameraRef.current.takePictureAsync({ base64: false });

          // Prepare FormData with RN-friendly file object
          const formData = new FormData();
          formData.append('image', {
            uri: photo.uri,
            name: 'photo.jpg',
            type: 'image/jpeg',
          });
          formData.append('user_prompt', query);
          formData.append('userPreferences', JSON.stringify(userPreferences));

          // Debug log
          for (let [key, value] of formData.entries()) console.log(key, value);

          const aiRes = await fetch(`http://${IP}:8000/askAi/`, {
            method: 'POST',
            body: formData,
          });

          if (!aiRes.ok) throw new Error(`AI request failed: ${aiRes.status}`);

          const aiData = await aiRes.json();
          const aiResponse = aiData.response;
          if (aiResponse) Speech.speak(aiResponse);
        } catch (err) {
          console.log("Camera capture or AI request failed:", err);
        }
      }
    };

    reader.readAsDataURL(audioBlob);
  } catch (e) {
    console.log('STT failed, using mock');

    setTimeout(async () => {
      const mockQuery = 'is the bus here';
      setConfirmingQuery(mockQuery);
      setIsAwaitingConfirmation(true);
      Speech.speak(`You said, "${mockQuery}". Do you want to proceed with this query?`);

      if (cameraRef.current) {
        try {
          const photo = await cameraRef.current.takePictureAsync({ base64: false });

          const formData = new FormData();
          formData.append('image', {
            uri: photo.uri,
            name: 'photo.jpg',
            type: 'image/jpeg',
          });
          formData.append('user_prompt', mockQuery);
          formData.append('userPreferences', JSON.stringify(userPreferences));

          for (let [key, value] of formData.entries()) console.log(key, value);

          const aiRes = await fetch(`http://${IP}:8000/askAi/`, {
            method: 'POST',
            body: formData,
          });

          const aiData = await aiRes.json();
          const aiResponse = aiData.response;
          if (aiResponse) Speech.speak(aiResponse);
        } catch (err) {
          console.log("Camera capture or AI request failed:", err);
        }
      }
    }, 500);
  }

  setIsProcessing(false);
  setRecording(null);
};




  // Insert this new function:
  const handleQuery = async (query) => {
    try {
      // 1. Check if the query requires visual analysis
      const needsVision = true; // Still assuming all transit queries need vision

      if (needsVision) {
        if (!permission?.granted) {
          await requestPermission();
          if (!permission?.granted) {
            Speech.speak("Camera permission is required to analyze the scene.");
            return; // Stop if permission is denied
          }
        }

        setShowCamera(true);
        Speech.speak('Hold steady, capturing in 3 seconds');

        // Wait for a few seconds to let the user stabilize the camera
        await new Promise(resolve => setTimeout(resolve, 3000));

        // --- REAL CAMERA CAPTURE LOGIC ---
        let imageBase64 = null;

        try {
          console.log("Taking actual photo...");
          const photo = await cameraRef.current.takePictureAsync({
            base64: true, // We need the Base64 data to send to the backend
            quality: 0.5, // Reducing quality saves time/bandwidth
          });
          imageBase64 = photo.base64;
        } catch (e) {
          console.error("Camera capture failed:", e);
          Speech.speak("I failed to take a picture. Please try again.");
          setShowCamera(false);
          return;
        }

        // 2. Send query, actual image, and user ID to the fusion endpoint
        try {
          const response = await axios.post(`http://${IP}:8000/fusion`, {
            userId: userId,
            query: query,
            image: imageBase64, // 👈 NOW SENDING REAL BASE64 DATA
          });

          const instruction = response.data.instruction;
          Speech.speak(instruction);

        } catch (e) {
          console.error("Fusion call failed:", e);
          Speech.speak(`I'm sorry, I couldn't connect to the core system. Please try again.`);
        }

        setShowCamera(false);
      } else {
        Speech.speak('All clear, the path is open.');
      }
    } catch (e) {
      console.error("Query error:", e);
      Alert.alert('Error', 'Query processing failed.');
    }
  };

  const confirmAndProceed = () => {
    // Only proceed if there is a query to confirm
    if (confirmingQuery) {
      // Clear confirmation state and pass the query to the main logic
      setIsAwaitingConfirmation(false);
      setConfirmingQuery(null);
      handleQuery(confirmingQuery);
    }
  };

  const cancelAndReanswer = () => {
    // Clear the current query and reset to the idle state
    setIsAwaitingConfirmation(false);
    setConfirmingQuery(null);
    Speech.speak("Query cancelled. Hold the microphone to speak again.");
    AccessibilityInfo.announceForAccessibility("Query cancelled. Hold the microphone to speak again.");
  };

  if (isAwaitingConfirmation) {
    return (
       <View style={styles.container}>

      {/* NO FULL-SCREEN PRESSABLE HERE */}
      <AnimatedGlow isRecording={isRecording || isProcessing} pulseAnim={pulseAnim} />

      {/* MAIN CONTENT */}
      <View style={styles.content}>
        <View style={styles.header}>
          <Text style={styles.geminiTitle}>Visionary Guide</Text>
          <Text style={styles.geminiSubtitle}>Your AI companion for navigation</Text>
        </View>

        <View style={styles.geminiCenter}>
          {/* 👇 ATTACH PRESSABLE TO THE MICROPHONE CIRCLE ONLY 👇 */}
          <Pressable
            onLongPress={startRecording}
            onPressOut={stopRecording} // ✅ CORRECT: Pass the function reference, NOT the result of a call
            // Setting a hit-slop ensures the touch area is forgiving even for a fast press-out
            hitSlop={20}
            disabled={isProcessing}
            accessible={true}
            accessibilityLabel={isRecording ? "Stop recording" : "Hold microphone button to start recording"}
          >
            {/* Microphone Icon Container */}
            <View
              style={[
                styles.geminiMicCircle,
                isRecording && styles.micRecording,
                isProcessing && styles.micProcessing
              ]}
            >
              {/* 🚨 ICON REPLACEMENT */}
              {isRecording ? (
                <Feather name="mic" size={80} color={styles.micIconListening.color} />
              ) : isProcessing ? (
                <Feather name="loader" size={80} color={styles.micProcessing.borderColor} style={styles.loaderAnimation} />
              ) : (
                <Feather name="zap" size={80} color={styles.geminiMicIcon.color} />
              )}
            </View>
          </Pressable>

          {/* Hint Text */}
          <Text style={[
            styles.geminiHint,
            isRecording && styles.hintListening
          ]}>
            {isRecording ? 'Listening...' : isProcessing ? 'Thinking...' : 'Hold microphone to speak'}
          </Text>
        </View>

        {/* Example Prompts Section */}
        <View style={styles.geminiExamples}>
          <Text style={styles.exampleText}>“Hey Visionary, is the bus here?”</Text>
          <Text style={styles.exampleText}>“What’s in front of me?”</Text>
          <Text style={styles.exampleText}>“Describe my surroundings”</Text>
        </View>
      </View>

      <Animated.View style={styles.animatedLoaderStyle} />
    </View>
    );
  }

  // --- CAMERA SCREEN (Unchanged) ---
  if (showCamera) {
    return (
      <View style={styles.cameraContainer}>
        <CameraView ref={cameraRef} style={StyleSheet.absoluteFillObject} facing="back" />
        <LinearGradient
          colors={['transparent', 'rgba(0,0,0,0.9)']}
          style={StyleSheet.absoluteFillObject}
          locations={[0.5, 1]}
        />
        <View style={styles.geminiCameraOverlay}>
          <Text style={styles.geminiAnalyzing}>Analyzing scene...</Text>
          <Text style={styles.geminiHold}>Hold Steady</Text>
        </View>
      </View>
    );
  }

  if (isPrefsLoading) {
    return (
      <View style={[styles.container, styles.geminiCenter]}>
        <ActivityIndicator size="large" color="#4285F4" />
        <Text style={styles.loadingText}>Loading user context...</Text>
      </View>
    );
  }

  // --- MAIN UI SCREEN (Fixed Logic) ---
   return (
    
    <View style={styles.cameraContainer}>
  <CameraView ref={cameraRef} style={StyleSheet.absoluteFillObject} facing="back" />
  <LinearGradient
          colors={['transparent', 'rgba(0,0,0,0.9)']}
          style={StyleSheet.absoluteFillObject}
          locations={[0.5, 1]}
        />
      {/* NO FULL-SCREEN PRESSABLE HERE */}
      <AnimatedGlow isRecording={isRecording || isProcessing} pulseAnim={pulseAnim} />

      {/* MAIN CONTENT */}
      <View style={styles.content}>
        <View style={styles.header}>
          <Text style={styles.geminiTitle}>Visionary Guide</Text>
          <Text style={styles.geminiSubtitle}>Your AI companion for navigation</Text>
        </View>

        <View style={styles.geminiCenter}>
          {/* 👇 ATTACH PRESSABLE TO THE MICROPHONE CIRCLE ONLY 👇 */}
          <Pressable
            onLongPress={startRecording}
            onPressOut={stopRecording} // ✅ CORRECT: Pass the function reference, NOT the result of a call
            // Setting a hit-slop ensures the touch area is forgiving even for a fast press-out
            hitSlop={20}
            disabled={isProcessing}
            accessible={true}
            accessibilityLabel={isRecording ? "Stop recording" : "Hold microphone button to start recording"}
          >
            {/* Microphone Icon Container */}
            <View
              style={[
                styles.geminiMicCircle,
                isRecording && styles.micRecording,
                isProcessing && styles.micProcessing
              ]}
            >
              {/* 🚨 ICON REPLACEMENT */}
              {isRecording ? (
                <Feather name="mic" size={80} color={styles.micIconListening.color} />
              ) : isProcessing ? (
                <Feather name="loader" size={80} color={styles.micProcessing.borderColor} style={styles.loaderAnimation} />
              ) : (
                <Feather name="zap" size={80} color={styles.geminiMicIcon.color} />
              )}
            </View>
          </Pressable>

          {/* Hint Text */}
          <Text style={[
            styles.geminiHint,
            isRecording && styles.hintListening
          ]}>
            {isRecording ? 'Listening...' : isProcessing ? 'Thinking...' : 'Hold microphone to speak'}
          </Text>
        </View>

        {/* Example Prompts Section */}
        <View style={styles.geminiExamples}>
          <Text style={styles.exampleText}>“Hey Visionary, is the bus here?”</Text>
          <Text style={styles.exampleText}>“What’s in front of me?”</Text>
          <Text style={styles.exampleText}>“Describe my surroundings”</Text>
        </View>
      </View>

      <Animated.View style={styles.animatedLoaderStyle} />
    </View>
  );
}

const styles = StyleSheet.create({
  // --- STYLES (Unchanged) ---
  container: {
    flex: 1,
    backgroundColor: '#1C1C1E',
  },
  camera: { flex: 1 },
  micOverlay: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
    alignItems: 'center',
  },
  glowWrapper: {
    position: 'absolute',
    width: 300,
    height: 300,
    alignSelf: 'center',
    top: height / 2 - 150,
    left: width / 2 - 150,
    borderRadius: 150,
    overflow: 'hidden',
    zIndex: 0,
  },
  glowGradient: {
    ...StyleSheet.absoluteFillObject,
    opacity: 0.9,
  },
  content: {
    flex: 1,
    justifyContent: 'space-between',
    paddingTop: 80,
    paddingBottom: 60,
    paddingHorizontal: 30,
    zIndex: 1,
  },
  header: {
    paddingTop: 12,
    paddingBottom: 8,
  },
  geminiTitle: {
    fontSize: 42,
    fontWeight: '900',
    color: 'white',
    textAlign: 'center',
    letterSpacing: -1,
  },
  geminiSubtitle: {
    fontSize: 18,
    color: '#A0A0A0',
    textAlign: 'center',
    marginTop: 10,
    fontWeight: '500',
  },
  geminiCenter: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  geminiMicCircle: {
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: 'rgba(255, 255, 255, 0.3)',
    shadowColor: 'rgba(255, 255, 255, 0.1)',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 1,
    shadowRadius: 10,
    elevation: 5,
  },
  micRecording: {
    borderColor: '#FB923C',
    shadowColor: '#FB923C',
    shadowRadius: 20,
  },
  micProcessing: {
    borderColor: '#FBBC05',
  },
  geminiMicIcon: {
    color: 'white',
  },
  micIconListening: {
    color: '#FB923C',
  },
  geminiHint: {
    marginTop: 30,
    fontSize: 18,
    color: '#A0A0A0',
    fontWeight: '600',
  },
  hintListening: {
    color: '#FB923C',
  },
  geminiExamples: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    padding: 20,
    borderRadius: 20,
  },
  exampleText: {
    color: 'white',
    fontSize: 16,
    marginBottom: 12,
    fontWeight: '500',
  },
  cameraContainer: {
    flex: 1,
    backgroundColor: 'black',
  },
  geminiCameraOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.4)',
  },
  geminiAnalyzing: {
    color: 'white',
    fontSize: 24,
    fontWeight: '800',
    marginBottom: 20,
  },
  geminiHold: {
    color: '#34A853',
    fontSize: 32,
    fontWeight: '900',
  },
  loaderAnimation: {
    transform: [{ rotate: '360deg' }],
    animation: 'spin 2s linear infinite',
  },
  animatedLoaderStyle: {
  },
  loadingText: { // 👈 NEW STYLE
    marginTop: 20,
    fontSize: 18,
    color: '#A0A0A0',
    fontWeight: '500',
  },
  confirmationTitle: {
    fontSize: 28,
    fontWeight: '900',
    color: 'white',
    marginBottom: 30,
  },
  queryBox: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    padding: 20,
    borderRadius: 15,
    width: '80%',
    alignItems: 'center',
    marginBottom: 30,
  },
  queryText: {
    fontSize: 20,
    color: '#4285F4',
    fontWeight: '600',
    textAlign: 'center',
  },
  confirmationPrompt: {
    fontSize: 20,
    color: '#A0A0A0',
    marginBottom: 40,
    fontWeight: '500',
  },
  buttonContainer: {
    flexDirection: 'row',
    width: '80%',
    justifyContent: 'space-between',
  },
  confirmButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 30,
    width: '48%',
  },
  proceedButton: {
    backgroundColor: '#34A853', // Google Green
  },
  cancelButton: {
    backgroundColor: '#EA4335', // Google Red
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '700',
    marginLeft: 10,
  },
});
