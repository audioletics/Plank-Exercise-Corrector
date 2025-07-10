import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class PlankDetector {
  static const String _modelPath = 'assets/ml/plank_model.tflite';
  static const String _scalerPath = 'assets/ml/plank_input_scaler.json';
  static const double _predictionThreshold = 0.6;

  late Interpreter _interpreter;
  late Map<String, dynamic> _scaler;
  
  String _previousStage = "unknown";
  List<Map<String, dynamic>> _results = [];
  bool _hasError = false;

  final List<String> _importantLandmarks = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
  ];

  Future<void> initialize() async {
    await _loadModel();
    await _loadScaler();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(_modelPath);
    } catch (e) {
      throw Exception('Error loading model: $e');
    }
  }

  Future<void> _loadScaler() async {
    try {
      final scalerData = await rootBundle.loadString(_scalerPath);
      _scaler = json.decode(scalerData);
    } catch (e) {
      throw Exception('Error loading scaler: $e');
    }
  }

  Map<String, dynamic> detect(List<Pose> poses, Uint8List imageBytes, int timestamp) {
    if (poses.isEmpty) {
      return {
        'stage': 'unknown',
        'probability': 0.0,
        'hasError': false,
      };
    }

    try {
      final keypoints = _extractKeypoints(poses.first);
      final scaledInput = _scaleInput(keypoints);
      final prediction = _predict(scaledInput);
      final result = _evaluatePrediction(prediction, imageBytes, timestamp);
      
      return result;
    } catch (e) {
      return {
        'stage': 'error',
        'probability': 0.0,
        'hasError': false,
        'error': e.toString(),
      };
    }
  }

  List<double> _extractKeypoints(Pose pose) {
    List<double> keypoints = [];
    
    for (String landmarkName in _importantLandmarks) {
      final landmark = _getLandmarkByName(pose, landmarkName);
      if (landmark != null) {
        keypoints.addAll([
          landmark.x,
          landmark.y,
          landmark.z ?? 0.0,
          landmark.visibility ?? 1.0,
        ]);
      } else {
        keypoints.addAll([0.0, 0.0, 0.0, 0.0]);
      }
    }
    
    return keypoints;
  }

  PoseLandmark? _getLandmarkByName(Pose pose, String name) {
    switch (name) {
      case "NOSE": return pose.landmarks[PoseLandmarkType.nose];
      case "LEFT_SHOULDER": return pose.landmarks[PoseLandmarkType.leftShoulder];
      case "RIGHT_SHOULDER": return pose.landmarks[PoseLandmarkType.rightShoulder];
      case "LEFT_ELBOW": return pose.landmarks[PoseLandmarkType.leftElbow];
      case "RIGHT_ELBOW": return pose.landmarks[PoseLandmarkType.rightElbow];
      case "LEFT_WRIST": return pose.landmarks[PoseLandmarkType.leftWrist];
      case "RIGHT_WRIST": return pose.landmarks[PoseLandmarkType.rightWrist];
      case "LEFT_HIP": return pose.landmarks[PoseLandmarkType.leftHip];
      case "RIGHT_HIP": return pose.landmarks[PoseLandmarkType.rightHip];
      case "LEFT_KNEE": return pose.landmarks[PoseLandmarkType.leftKnee];
      case "RIGHT_KNEE": return pose.landmarks[PoseLandmarkType.rightKnee];
      case "LEFT_ANKLE": return pose.landmarks[PoseLandmarkType.leftAnkle];
      case "RIGHT_ANKLE": return pose.landmarks[PoseLandmarkType.rightAnkle];
      case "LEFT_HEEL": return pose.landmarks[PoseLandmarkType.leftHeel];
      case "RIGHT_HEEL": return pose.landmarks[PoseLandmarkType.rightHeel];
      case "LEFT_FOOT_INDEX": return pose.landmarks[PoseLandmarkType.leftFootIndex];
      case "RIGHT_FOOT_INDEX": return pose.landmarks[PoseLandmarkType.rightFootIndex];
      default: return null;
    }
  }

  List<double> _scaleInput(List<double> input) {
    // Implementasi scaling sederhana - sesuaikan dengan scaler Anda
    return input.map((value) => (value - 0.5) * 2.0).toList();
  }

  List<double> _predict(List<double> input) {
    final inputTensor = [input];
    final outputTensor = List.filled(1 * 3, 0.0).reshape([1, 3]);
    
    _interpreter.run(inputTensor, outputTensor);
    
    return outputTensor[0];
  }

  Map<String, dynamic> _evaluatePrediction(
    List<double> prediction, 
    Uint8List imageBytes, 
    int timestamp
  ) {
    final maxIndex = prediction.indexOf(prediction.reduce((a, b) => a > b ? a : b));
    final maxProb = prediction[maxIndex];
    
    String currentStage;
    if (maxProb >= _predictionThreshold) {
      switch (maxIndex) {
        case 0: currentStage = "correct"; break;
        case 1: currentStage = "low back"; break;
        case 2: currentStage = "high back"; break;
        default: currentStage = "unknown";
      }
    } else {
      currentStage = "unknown";
    }

    if (currentStage == "low back" || currentStage == "high back") {
      if (_previousStage != currentStage) {
        _results.add({
          'stage': currentStage,
          'imageBytes': imageBytes,
          'timestamp': timestamp,
        });
        _hasError = true;
      }
    } else {
      _hasError = false;
    }

    _previousStage = currentStage;

    return {
      'stage': currentStage,
      'probability': maxProb,
      'hasError': _hasError,
      'results': _results,
    };
  }

  void clearResults() {
    _previousStage = "unknown";
    _results.clear();
    _hasError = false;
  }

  void dispose() {
    _interpreter.close();
  }
}