import UIKit

class ViewController: UIViewController {
    @IBOutlet weak var uiOutput: UILabel!
    @IBOutlet weak var uiButton: UIButton!
    @IBOutlet weak var uiPrompt: UITextField!
    @IBOutlet weak var uiTemperature: UISlider!
    @IBOutlet weak var uiTopp: UISlider!
    
    let llama2 = Llama2()
    var tokenizer: Tokenizer!
    var generating: Bool?
    var tokens: [Int32] = []
    private var workItem: DispatchWorkItem?
    
    func initTokenizer() {
        let tokenizerPath = Bundle.main.path(forResource: "tokenizer", ofType: "bin")
        tokenizer = Tokenizer(tokenizerPath:tokenizerPath)!
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        initTokenizer()
        
        generating = false
        tokens = []
    }

    @IBAction func btnClicked(_ sender: Any) {
        if(!generating!) {
            let prompt = uiPrompt.text!
            let temperature = uiTemperature.value
            let topp = uiTopp.value
            
            tokens = []
            tokens.append(1)
            
            if (prompt.count > 0) {
                let promptTokens = tokenizer.tokenize(prompt)
                for i in 0..<promptTokens!.count {
                    tokens.append(promptTokens![i].int32Value)
                }
            }
            
            //Clear output
            self.setText(text: prompt)
            
            //Start generating thread
            workItem = DispatchWorkItem {
              while(!self.workItem!.isCancelled) {
                  if(self.tokens.count < 256) {
                      DispatchQueue.main.sync {
                          let logits = self.llama2.predict(prompt: self.tokens)
                          let nextToken = logits.withUnsafeBufferPointer { (buffer) -> Int32 in
                              let p = buffer.baseAddress
                              return Int32(self.tokenizer.sample(p!, temperature, topp))
                          }
                          self.tokens.append(nextToken)
                          
                          var nextText = self.tokenizer.get_result(nextToken)
                          if(nextText == "<0x0A>") {
                              nextText = "\n"
                          }
                          
                          var text = self.uiOutput.text
                          text! += nextText!
                          
                          self.setText(text: text!)
                      }
                  } else {
                      break
                  }
              }
            }
            DispatchQueue.global().async(execute: workItem!)

            generating = true
            uiButton.setTitle("Stop", for: UIControl.State.normal)
        } else {
            //Stop generating thread
            workItem!.cancel()

            generating = false
            uiButton.setTitle("Start", for: UIControl.State.normal)
        }
    }
    
    @objc func setText(text: String) {
        uiOutput.text = text
    }
}

