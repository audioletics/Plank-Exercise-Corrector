import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:math';

class Vector3 {
  double x, y, z;
  Vector3(this.x, this.y, this.z);

  Vector3 operator -(Vector3 other) => Vector3(x - other.x, y - other.y, z - other.z);
  Vector3 operator /(double scalar) => Vector3(x / scalar, y / scalar, z / scalar);
  double norm() => sqrt(x * x + y * y + z * z);
}

class PlankDetector {
  static const String _modelPath = 'assets/ml/plank_model_norm_60.tflite';
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

  


  Map<String, dynamic> detect(List<Pose> poses, Uint8List imageBytes, int timestamp) {
    if (poses.isEmpty) {
      return {'stage': 'unknown', 'probability': 0.0, 'hasError': false};
    }
    try {
      // 1. Panggil fungsi pre-processing yang baru
      final features = _processPoseFeatures(poses.first);

      // 2. Jika fitur tidak bisa dibuat (pose tidak lengkap), kembalikan status 'unknown'
      if (features == null) {
        return {'stage': 'unknown', 'probability': 0.0, 'hasError': false};
      }
      
      // 3. Masukkan fitur ke model untuk prediksi
      final prediction = _predict(features);
      
      // 4. Evaluasi hasil prediksi
      final result = _evaluatePrediction(prediction, imageBytes, timestamp);
      return result;

    } catch (e) {
      print('ERROR TERJADI DI DALAM DETECT: $e');
      return {'stage': 'error', 'probability': 0.0, 'hasError': false, 'error': e.toString()};
    }
  }


  List<double>? _processPoseFeatures(Pose pose) {
    
    // --- Langkah A: Ekstrak koordinat mentah ke dalam objek Vector3 ---
    final Map<PoseLandmarkType, Vector3> coords = {};
    pose.landmarks.forEach((type, landmark) {
      coords[type] = Vector3(landmark.x, landmark.y, landmark.z);
    });

    // Cek apakah landmark kunci untuk normalisasi terdeteksi
    final requiredForNorm = [PoseLandmarkType.leftHip, PoseLandmarkType.rightHip, PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder];
    for (var type in requiredForNorm) {
      if (coords[type] == null) {
        print('Preprocessing Gagal: Landmark ${type.name} untuk normalisasi tidak ditemukan.');
        return null; 
      }
    }

    // --- Langkah B: Normalisasi Posisi (Translation) ---
    final leftHip = coords[PoseLandmarkType.leftHip]!;
    final rightHip = coords[PoseLandmarkType.rightHip]!;
    final hipCenter = Vector3((leftHip.x + rightHip.x) / 2, (leftHip.y + rightHip.y) / 2, (leftHip.z + rightHip.z) / 2);
    
    final Map<PoseLandmarkType, Vector3> translatedCoords = {};
    coords.forEach((type, landmark) {
      translatedCoords[type] = landmark - hipCenter;
    });
    
    // --- Langkah C: Normalisasi Skala ---
    final leftShoulder = translatedCoords[PoseLandmarkType.leftShoulder]!;
    final rightShoulder = translatedCoords[PoseLandmarkType.rightShoulder]!;
    final shoulderCenter = Vector3((leftShoulder.x + rightShoulder.x) / 2, (leftShoulder.y + rightShoulder.y) / 2, (leftShoulder.z + rightShoulder.z) / 2);
    
    final torsoSize = shoulderCenter.norm();
    if (torsoSize < 1e-6) {
      print('Preprocessing Gagal: Ukuran torso tidak valid.');
      return null;
    }

    final Map<PoseLandmarkType, Vector3> finalCoords = {};
    translatedCoords.forEach((type, landmark) {
      finalCoords[type] = landmark / torsoSize;
    });

    // --- Langkah D: Hitung Fitur Sudut ---
    final Map<String, double> angles = {};
    double
    calculateAngle(PoseLandmarkType a, PoseLandmarkType b, PoseLandmarkType c) {
      if (finalCoords[a] != null && finalCoords[b] != null && finalCoords[c] != null) {
        return _calculateAngle(finalCoords[a]!, finalCoords[b]!, finalCoords[c]!);
      }
      return 0.0; // Kembalikan 0 jika salah satu landmark tidak ada
    }

    angles['L_ELBOW'] = calculateAngle(PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist);
    angles['R_ELBOW'] = calculateAngle(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist);
    angles['L_SHOULDER'] = calculateAngle(PoseLandmarkType.leftHip, PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow);
    angles['R_SHOULDER'] = calculateAngle(PoseLandmarkType.rightHip, PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow);
    angles['L_HIP'] = calculateAngle(PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee);
    angles['R_HIP'] = calculateAngle(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee);
    angles['L_KNEE'] = calculateAngle(PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle);
    angles['R_KNEE'] = calculateAngle(PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle);
    angles['BACK'] = calculateAngle(PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip, PoseLandmarkType.leftAnkle);

    // --- Langkah E: Gabungkan Semua Fitur menjadi satu List ---
    final List<double> featureRow = [];
    final landmarksToProcess = [
      PoseLandmarkType.nose, PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder, PoseLandmarkType.leftElbow, PoseLandmarkType.rightElbow,
      PoseLandmarkType.leftWrist, PoseLandmarkType.rightWrist, PoseLandmarkType.leftHip, PoseLandmarkType.rightHip, PoseLandmarkType.leftKnee,
      PoseLandmarkType.rightKnee, PoseLandmarkType.leftAnkle, PoseLandmarkType.rightAnkle, PoseLandmarkType.leftHeel, PoseLandmarkType.rightHeel,
      PoseLandmarkType.leftFootIndex, PoseLandmarkType.rightFootIndex,
    ];

    // 1. Tambahkan 51 fitur koordinat (17 landmark * 3)
    for (var type in landmarksToProcess) {
      final coord = finalCoords[type] ?? Vector3(0,0,0); // Jika landmark tidak ada, gunakan (0,0,0)
      featureRow.addAll([coord.x, coord.y, coord.z]);
    }
    // 2. Tambahkan 9 fitur sudut
    featureRow.addAll(angles.values);
    
    return featureRow; // Total 60 fitur
  }

  double _calculateAngle(Vector3 a, Vector3 b, Vector3 c) {
    // ... (Fungsi ini tidak berubah) ...
    final vecBA = a - b;
    final vecBC = c - b;
    final dotProduct = (vecBA.x * vecBC.x) + (vecBA.y * vecBC.y) + (vecBA.z * vecBC.z);
    final normBA = vecBA.norm();
    final normBC = vecBC.norm();
    if (normBA == 0 || normBC == 0) return 0.0;
    var cosineAngle = dotProduct / (normBA * normBC);
    cosineAngle = cosineAngle.clamp(-1.0, 1.0);
    final angle = acos(cosineAngle);
    return angle * (180 / pi);
  }

  List<double> _extractKeypoints(Pose pose, Size imageSize, InputImageRotation rotation) {
    List<double> keypoints = [];
    for (final landmarkType in _importantLandmarks) {
      final landmark = pose.landmarks[landmarkType];
      if (landmark != null) {
        double dx = landmark.x;
        double dy = landmark.y;
        double normalizedX;
        double normalizedY;

        // Logika untuk handle rotasi
        switch (rotation) {
          case InputImageRotation.rotation90deg:
          case InputImageRotation.rotation270deg:
            // Saat gambar dirotasi 90/270 (portrait), sumbu x dan y tertukar
            normalizedX = dy / imageSize.height;
            normalizedY = dx / imageSize.width;
            break;
          default: // Termasuk rotation0deg dan rotation180deg (landscape)
            normalizedX = dx / imageSize.width;
            normalizedY = dy / imageSize.height;
        }

        keypoints.addAll([
          normalizedX,
          normalizedY,
          landmark.z,
          landmark.likelihood,
        ]);

        print('Rotation: $rotation | Raw: (${dx.toStringAsFixed(2)}, ${dy.toStringAsFixed(2)}) | Norm: (${normalizedX.toStringAsFixed(2)}, ${normalizedY.toStringAsFixed(2)})');


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
        case 1: currentStage = "high back"; break;
        case 2: currentStage = "low back"; break;
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