

import UIKit
import AVFoundation
class ViewController: UIViewController {
    //Capture selection
    var session: AVCaptureSession?
    //Photo selection
    let output = AVCapturePhotoOutput()
    //Video preview
    let previewLayer = AVCaptureVideoPreviewLayer()
    //Shulter button
    private let shutterButton: UIButton = {
        let button = UIButton(frame: CGRect(x:  0, y: 0, width: 100, height: 100))
        button.layer.cornerRadius = 50
        button.layer.borderWidth = 10
        button.layer.borderColor = UIColor.white.cgColor
        return button
    }()
    
    
    override func viewDidLoad(){
        super.viewDidLoad()
        view.backgroundColor = .black
        view.layer.addSublayer(previewLayer)
        view.addSubview(shutterButton)
        checkCameraPermssions()
        
        shutterButton.addTarget(self, action: #selector(didTapTakephoto), for:.touchUpInside)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        
        shutterButton.center = CGPoint(x: view.frame.width/2,
                                       y: view.frame.size.height - 100)
        
    }
    
    private func checkCameraPermssions(){
        switch AVCaptureDevice.authorizationStatus(for: .video){
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video){ [weak self] granted in
                guard granted else{
                    return
                }
                DispatchQueue.main.async{
                    self?.setUpCamera()
                }
                
            }
        case.restricted:
            break
        case .denied:
            break
        case.authorized:
            setUpCamera()
        @unknown default:
            break
        }
    }
    
    private func setUpCamera(){
        let session = AVCaptureSession()
        if let device = AVCaptureDevice.default(for: .video){
            do{
                let input = try AVCaptureDeviceInput(device: device)
                if session.canAddInput(input){
                    session.addInput(input)
                }
                if session.canAddOutput(output){
                    session.addOutput(output)
                }
                
                previewLayer.videoGravity = .resizeAspectFill
                previewLayer.session = session
                
                session.startRunning()
                self.session = session
            }
            catch{
                print(error)
                
            }
        }
    }
    @objc private func didTapTakephoto(){
        output.capturePhoto(with: AVCapturePhotoSettings(),
                            delegate: self)
    }
}

extension ViewController: AVCapturePhotoCaptureDelegate{
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard let data = photo.fileDataRepresentation() else{
            return
        }
        let image = UIImage(data: data)
        
        //session?.stopRunning()
        
        let imageView = UIImageView(image: image)
        imageView.contentMode = .scaleAspectFill
        imageView.frame = view.bounds
        view.addSubview(imageView)
        
        DispatchQueue.main.asyncAfter(deadline: .now() + .seconds(2), execute: {
            imageView.removeFromSuperview()
        })
        
        
        
        
    }
    
}
