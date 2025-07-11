import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import 'detector_view.dart';
import 'painters/plank_pose_painter.dart';
import 'plank_detector.dart';

class PlankDetectorView extends StatefulWidget {
  @override
  State<PlankDetectorView> createState() => _PlankDetectorViewState();
}

class _PlankDetectorViewState extends State<PlankDetectorView> {
  final PoseDetector _poseDetector = PoseDetector(
    options: PoseDetectorOptions(
      model: PoseDetectionModel.accurate,
      mode: PoseDetectionMode.stream,
    ),
  );
  
  late PlankDetector _plankDetector;
  bool _canProcess = true;
  bool _isBusy = false;
  CustomPaint? _customPaint;
  String? _text;
  var _cameraLensDirection = CameraLensDirection.back;

  @override
  void initState() {
    super.initState();
    _initializePlankDetector();
  }

  Future<void> _initializePlankDetector() async {
    _plankDetector = PlankDetector();
    try {
      await _plankDetector.initialize();
    } catch (e) {
      print('Error initializing plank detector: $e');
    }
  }

  @override
  void dispose() {
    _canProcess = false;
    _poseDetector.close();
    _plankDetector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(children: [
        DetectorView(
          title: 'Plank Exercise Corrector',
          customPaint: _customPaint,
          text: _text,
          onImage: _processImage,
          initialCameraLensDirection: _cameraLensDirection,
          onCameraLensDirectionChanged: (value) => _cameraLensDirection = value,
          initialDetectionMode: DetectorViewMode.liveFeed,
          onDetectorViewModeChanged: _onScreenModeChanged,
        ),
        _buildControlPanel(),
      ]),
    );
  }

  Widget _buildControlPanel() {
    return Positioned(
      top: 30,
      left: 20,
      right: 20,
      child: Container(
        padding: EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(10),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            ElevatedButton(
              onPressed: _clearResults,
              child: Text('Clear Results'),
            ),
            ElevatedButton(
              onPressed: _showResults,
              child: Text('Show Results'),
            ),
          ],
        ),
      ),
    );
  }

  void _onScreenModeChanged(DetectorViewMode mode) {
    // Handle mode changes if needed
  }

  Future<void> _processImage(InputImage inputImage) async {
    if (!_canProcess) return;
    if (_isBusy) return;
    _isBusy = true;

    setState(() {
      _text = '';
    });

    final poses = await _poseDetector.processImage(inputImage);

    final imageSize = inputImage.metadata?.size;
    final imageRotation = inputImage.metadata?.rotation; 
    
    if (poses.isNotEmpty && inputImage.bytes != null) {
      final result = _plankDetector.detect(
        poses, 
        imageSize!,
        imageRotation!,
        inputImage.bytes!, 
        DateTime.now().millisecondsSinceEpoch,
      );

      // --- PERBAIKAN: Cek apakah ada key 'error' pada hasil deteksi ---
      if (result.containsKey('error')) {
        // Jika ada error, tampilkan pesan error-nya
        _text = 'Error: ${result['error']}';
        _customPaint = null; // Kosongkan painter
      } else if (inputImage.metadata?.size != null &&
          inputImage.metadata?.rotation != null) {
        final painter = PlankPosePainter(
          poses,
          inputImage.metadata!.size,
          inputImage.metadata!.rotation,
          _cameraLensDirection,
          result['hasError'] as bool,
          result['stage'] as String,
          result['probability'] as double,
        );
        _customPaint = CustomPaint(painter: painter);
      } else {
        _text = 'Stage: ${result['stage']}\n'
            'Probability: ${(result['probability'] as double).toStringAsFixed(2)}\n'
            'Poses detected: ${poses.length}';
        _customPaint = null;
      }
    } else {
      _text = 'No poses detected';
      _customPaint = null;
    }

    _isBusy = false;
    if (mounted) {
      setState(() {});
    }
  }


  void _clearResults() {
    _plankDetector.clearResults();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Results cleared')),
    );
  }

  void _showResults() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Exercise Results'),
        content: Text('Results feature coming soon'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('OK'),
          ),
        ],
      ),
    );
  }
}