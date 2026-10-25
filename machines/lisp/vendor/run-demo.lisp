;;;; run-demo.lisp
;;;; Run a full demonstration of the Lisp Cognito Philosopher

(format t "~%~%")
(format t "╔════════════════════════════════════════════════════════════╗~%")
(format t "║    Lisp Cognito Philosopher - Live Demonstration          ║~%")
(format t "╚════════════════════════════════════════════════════════════╝~%")
(format t "~%")

;; Load all source files
(format t "Loading system...~%")
(load "package.lisp")
(load "utils.lisp")
(load "goedel-encoder.lisp")
(load "symbolic-operator.lisp")
(load "geometric-tokenizer.lisp")
(load "cognito-construct.lisp")
(load "dialectic-macros.lisp")
(load "living-logos.lisp")
(load "reflection-engine.lisp")
(load "philosopher.lisp")
(format t "✅ System loaded~%~%")

(in-package :lisp-cognito-philosopher)

;; Set log level to info
(setf *log-level* :info)

(format t "╔════════════════════════════════════════════════════════════╗~%")
(format t "║  Part 1: Creating a Philosopher                            ║~%")
(format t "╚════════════════════════════════════════════════════════════╝~%~%")

(defparameter *philosopher* 
  (make-philosopher "What is the nature of consciousness and self-reference?"))

(format t "~%╔════════════════════════════════════════════════════════════╗~%")
(format t "║  Part 2: Philosophical Contemplation (10 cycles)           ║~%")
(format t "╚════════════════════════════════════════════════════════════╝~%~%")

(dotimes (cycle 10)
  (format t "~%--- Cycle ~D/10 ---~%" (1+ cycle))
  
  ;; Think about philosophical concepts
  (think *philosopher* "consciousness")
  (think *philosopher* "self-reference")
  (think *philosopher* "emergence")
  
  ;; Reflect and show gap
  (let ((reflection (reflect *philosopher*)))
    (format t "Gödel Gap: ~,3F (~A)~%" 
            (getf reflection :gap)
            (getf reflection :interpretation))
    (format t "Diagnosis: ~A~%" (getf reflection :diagnosis)))
  
  ;; Self-correct if needed
  (when (> (get-goedel-gap *philosopher*) 0.4)
    (format t "→ Applying self-correction~%")
    (self-correct *philosopher*)))

(format t "~%~%╔════════════════════════════════════════════════════════════╗~%")
(format t "║  Part 3: Generating Philosophical Text                    ║~%")
(format t "╚════════════════════════════════════════════════════════════╝~%")

(let ((constructs (list (make-living-construct :emergence)
                       (make-living-construct :information)
                       (make-living-construct :consciousness)
                       (make-living-construct :transcendence))))
  (let ((text (generate-philosophical-text constructs '((x . 1.0)) 5)))
    (format t "~A" text)))

(format t "~%╔════════════════════════════════════════════════════════════╗~%")
(format t "║  Part 4: Dialectic Self-Modification                      ║~%")
(format t "╚════════════════════════════════════════════════════════════╝~%~%")

;; Define a thesis about consciousness
(defthesis consciousness-thesis ()
  (list 'multiply
        (list 'log (list 'add (list :var 'x) 1))
        (list 'sin (list :var 'x))))

;; Define antithesis (negation)
(defantithesis consciousness-antithesis (thesis-result)
  (list 'exp (list 'negate thesis-result)))

;; Create synthesis
(defsynthesis consciousness-synthesis consciousness-thesis consciousness-antithesis)

(format t "Original Synthesis: ~A~%" (ast-to-string (consciousness-synthesis)))

;; Self-modify the thesis
(format t "~%Applying self-modification...~%")
(modify-generator "CONSCIOUSNESS-THESIS" 
  (list 'add
        (list 'cos (list :var 'x))
        (list 'sqrt (list 'add (list :var 'x) 1))))

(format t "Modified Synthesis: ~A~%~%" (ast-to-string (consciousness-synthesis)))

(format t "~%╔════════════════════════════════════════════════════════════╗~%")
(format t "║  Part 5: Final State Report                               ║~%")
(format t "╚════════════════════════════════════════════════════════════╝~%~%")

(format t "Total Thoughts: ~D~%" (cognito-construct-thought-count *philosopher*))
(format t "Current Gödel Gap: ~,3F~%" (get-goedel-gap *philosopher*))
(format t "Average Gap (last 10): ~,3F~%" (get-average-gap *philosopher* 10))
(format t "Gap Interpretation: ~A~%" (interpret-gap (get-goedel-gap *philosopher*)))

(format t "~%~%")
(format t "╔════════════════════════════════════════════════════════════╗~%")
(format t "║              Demonstration Complete!                       ║~%")
(format t "║                                                            ║~%")
(format t "║  The Lisp Cognito Philosopher has:                        ║~%")
(format t "║  • Contemplated philosophical questions                   ║~%")
(format t "║  • Measured its own self-knowledge (Gödel Gap)            ║~%")
(format t "║  • Generated philosophical text from living constructs    ║~%")
(format t "║  • Modified its own reasoning patterns                    ║~%")
(format t "║                                                            ║~%")
(format t "║  This demonstrates self-referential reasoning in action!  ║~%")
(format t "╚════════════════════════════════════════════════════════════╝~%")
(format t "~%~%")

;; Exit
(sb-ext:exit :code 0)
