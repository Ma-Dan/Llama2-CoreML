import Foundation
import UIKit
import CoreML

class Llama2 {
    var model: llama2
    var input: MLMultiArray
    var input_ptr: UnsafeMutablePointer<Float>
    
    public init() {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        model = try! llama2(configuration: config)
                       
        input = try! MLMultiArray(shape: [1, 256], dataType: .float32)
        input_ptr = UnsafeMutablePointer<Float>(OpaquePointer(input.dataPointer))
    }
    
    public func predict(prompt: [Int32]) -> [Float32] {
        let offset = prompt.count-1
        for i in 0..<input.shape[1].intValue {
            if (i < prompt.count) {
                input_ptr[i] = Float(prompt[i])
            } else {
                input_ptr[i] = 0.0
            }
        }
        
        let output = try? model.prediction(tokens: input)
        
        var result = [Float32]()
        let logits = output!.var_896
        let length = logits.shape[2].intValue
        for i in 0..<length {
            result.append(logits[offset*length+i].floatValue)
        }
        
        return result
    }
}
