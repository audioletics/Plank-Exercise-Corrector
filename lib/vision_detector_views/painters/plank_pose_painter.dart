import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import 'coordinates_translator.dart';

class PlankPosePainter extends CustomPainter {
  PlankPosePainter(
    this.poses,
    this.imageSize,
    this.rotation,
    this.cameraLensDirection,
    this.hasError,
    this.stage,
    this.probability,
  );

  final List<Pose> poses;
  final Size imageSize;
  final InputImageRotation rotation;
  final CameraLensDirection cameraLensDirection;
  final bool hasError;
  final String stage;
  final double probability;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..color = hasError ? Colors.red : Colors.green;

    final leftPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = hasError ? Colors.red : Colors.green;

    final rightPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = hasError ? Colors.red : Colors.green;

    for (final pose in poses) {
      pose.landmarks.forEach((_, landmark) {
        canvas.drawCircle(
          Offset(
            translateX(
              landmark.x,
              size,
              imageSize,
              rotation,
              cameraLensDirection,
            ),
            translateY(
              landmark.y,
              size,
              imageSize,
              rotation,
              cameraLensDirection,
            ),
          ),
          1,
          paint,
        );
      });

      void paintLine(
          PoseLandmarkType type1, PoseLandmarkType type2, Paint paintType) {
        final PoseLandmark joint1 = pose.landmarks[type1]!;
        final PoseLandmark joint2 = pose.landmarks[type2]!;
        canvas.drawLine(
          Offset(
            translateX(
              joint1.x,
              size,
              imageSize,
              rotation,
              cameraLensDirection,
            ),
            translateY(
              joint1.y,
              size,
              imageSize,
              rotation,
              cameraLensDirection,
            ),
          ),
          Offset(
            translateX(
              joint2.x,
              size,
              imageSize,
              rotation,
              cameraLensDirection,
            ),
            translateY(
              joint2.y,
              size,
              imageSize,
              rotation,
              cameraLensDirection,
            ),
          ),
          paintType,
        );
      }

      // Paint pose connections
      paintLine(PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder, paint);
      paintLine(PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow, leftPaint);
      paintLine(PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist, leftPaint);
      paintLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow, rightPaint);
      paintLine(PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist, rightPaint);
      paintLine(PoseLandmarkType.leftHip, PoseLandmarkType.rightHip, paint);
      paintLine(PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip, leftPaint);
      paintLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip, rightPaint);
      paintLine(PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee, leftPaint);
      paintLine(PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle, leftPaint);
      paintLine(PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee, rightPaint);
      paintLine(PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle, rightPaint);
    }

    _drawStatusBox(canvas, size);
  }

  void _drawStatusBox(Canvas canvas, Size size) {
    final rect = Rect.fromLTWH(10, 10, 250, 60);
    final paint = Paint()..color = Color(0xFFF57510);
    canvas.drawRect(rect, paint);

    final textPainter = TextPainter(
      text: TextSpan(
        children: [
          TextSpan(
            text: 'PROB: ${probability.toStringAsFixed(2)}\n',
            style: TextStyle(color: Colors.white, fontSize: 12),
          ),
          TextSpan(
            text: 'CLASS: $stage',
            style: TextStyle(color: Colors.white, fontSize: 12),
          ),
        ],
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(canvas, Offset(15, 15));
  }

  @override
  bool shouldRepaint(covariant PlankPosePainter oldDelegate) {
    return oldDelegate.imageSize != imageSize || oldDelegate.poses != poses;
  }
}