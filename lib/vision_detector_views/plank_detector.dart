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

  // --- PERBAIKAN: Gunakan enum langsung untuk efisiensi dan type safety ---
  final List<PoseLandmarkType> _importantLandmarks = const [
    PoseLandmarkType.nose,
    PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder,
    PoseLandmarkType.leftElbow, PoseLandmarkType.rightElbow,
    PoseLandmarkType.leftWrist, PoseLandmarkType.rightWrist,
    PoseLandmarkType.leftHip, PoseLandmarkType.rightHip,
    PoseLandmarkType.leftKnee, PoseLandmarkType.rightKnee,
    PoseLandmarkType.leftAnkle, PoseLandmarkType.rightAnkle,
    PoseLandmarkType.leftHeel, PoseLandmarkType.rightHeel,
    PoseLandmarkType.leftFootIndex, PoseLandmarkType.rightFootIndex,
  ];

  Future<void> initialize() async {
    await _loadModel();
    // --- PERBAIKAN: Aktifkan kembali pemuatan scaler ---
    await _loadScaler();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(_modelPath);
    } catch (e) {
      throw Exception('Error loading model: $e');
    }
  }

  // --- PERBAIKAN: Aktifkan dan implementasikan _loadScaler ---
  Future<void> _loadScaler() async {
    try {
      final scalerData = await rootBundle.loadString(_scalerPath);
      _scaler = json.decode(scalerData);
    } catch (e) {
      throw Exception('Error loading scaler: $e');
    }
  }

  


  Map<String, dynamic> detect(List<Pose> poses, Size imageSize, Uint8List imageBytes, int timestamp) {
    if (poses.isEmpty) {
      return {'stage': 'unknown', 'probability': 0.0, 'hasError': false};
    }

    try {
      final keypoints = _extractKeypoints(poses.first, imageSize);
      // --- PERBAIKAN: Terapkan scaling pada input ---
      // TAMBAHKAN PRINT DI SINI
      print('Jumlah Keypoints: ${keypoints.length}'); // Harusnya 68

      //  No need to scale input the model expects raw keypoints
      // final scaledInput = _scaleInput(keypoints);

      // // TAMBAHKAN PRINT LAGI DI SINI
      print('Input ke Model (Sudah Normalisasi): $keypoints'); // Cek apakah ada nilai aneh seperti NaN atau Infinity

      final prediction = _predict(keypoints);
      final result = _evaluatePrediction(prediction, imageBytes, timestamp);

      return result;
    } catch (e) {
      print('ERROR TERJADI DI DALAM DETECT: $e');
      return {'stage': 'error', 'probability': 0.0, 'hasError': false, 'error': e.toString()};
    }
  }

  List<double> _extractKeypoints(Pose pose, Size imageSize) {
    List<double> keypoints = [];
    for (final landmarkType in _importantLandmarks) {
      final landmark = pose.landmarks[landmarkType];
      if (landmark != null) {
        keypoints.addAll([
          // Lakukan normalisasi di sini
          landmark.x / imageSize.width,
          landmark.y / imageSize.height,
          landmark.z, // z dan likelihood tidak perlu dinormalisasi
          landmark.likelihood,
        ]);
      } else {
        keypoints.addAll([0.0, 0.0, 0.0, 0.0]);
      }
    }
    return keypoints;
  }

  // --- PERBAIKAN: Hapus fungsi _getLandmarkByName karena tidak lagi dibutuhkan ---

  // --- PERBAIKAN: Implementasi logic untuk scaling input ---
  List<double> _scaleInput(List<double> input) {
    final mean = List<double>.from(_scaler['mean']);
    final scale = List<double>.from(_scaler['scale']);
    List<double> scaledInput = List<double>.filled(input.length, 0.0);

    for (int i = 0; i < input.length; i++) {
      // Rumus StandardScaler: z = (x - u) / s
      // Tambahkan pengecekan untuk menghindari pembagian dengan nol
      if (scale[i] != 0) {
        scaledInput[i] = (input[i] - mean[i]) / scale[i];
      } else {
        scaledInput[i] = input[i] - mean[i];
      }
    }
    return scaledInput;
  }

  List<double> _predict(List<double> input) {
    // Input shape [1, 68] -> 17 landmark * 4 value (x, y, z, likelihood)
    final inputTensor = [input];
    // Output shape [1, 3] -> 3 kelas (correct, low, high)
    final outputTensor = List.filled(1 * 3, 0.0).reshape([1, 3]);

    _interpreter.run(inputTensor, outputTensor);

    return outputTensor[0];
  }

  Map<String, dynamic> _evaluatePrediction(List<double> prediction, Uint8List imageBytes, int timestamp) {
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
        _results.add({'stage': currentStage, 'imageBytes': imageBytes, 'timestamp': timestamp});
        _hasError = true;
      }
    } else {
      _hasError = false;
    }

    _previousStage = currentStage;

    return {'stage': currentStage, 'probability': maxProb, 'hasError': _hasError, 'results': _results};
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