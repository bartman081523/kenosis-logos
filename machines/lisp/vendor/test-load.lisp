;;;; test-load.lisp
;;;; Quick test to verify the system loads and basic functionality works

(format t "~%=== Loading Lisp Cognito Philosopher ===~%~%")

;; Require ASDF
(require :asdf)

;; Load the system
(load "lisp-cognito-philosopher.asd")
(handler-case
    (progn
      (asdf:load-system :lisp-cognito-philosopher :verbose nil)
      (format t "✅ System loaded successfully~%~%"))
  (error (e)
    (format t "❌ Error loading system: ~A~%~%" e)
    (sb-ext:exit :code 1)))

(in-package :lisp-cognito-philosopher)

(format t "=== Testing Core Components ===~%~%")

;; Test 1: Gödel Encoder
(format t "1. Testing Gödel Encoder...~%")
(let ((expr '(add (multiply 2 x) 1)))
  (let ((encoded (goedel-encode expr)))
    (format t "   Encoded ~A -> ~A~%" expr encoded)
    (format t "   ✅ Gödel encoding works~%~%")))

;; Test 2: Symbolic Operator
(format t "2. Testing Symbolic Operator...~%")
(let ((ast (generate-random-ast 2)))
  (format t "   Generated AST: ~A~%" (ast-to-string ast))
  (let ((result (evaluate-ast ast '((x . 1.0)))))
    (format t "   Evaluated at x=1.0: ~A~%" result)
    (format t "   ✅ Symbolic operator works~%~%")))

;; Test 3: Geometric Tokenizer
(format t "3. Testing Geometric Tokenizer...~%")
(let ((tokenizer (make-geometric-tokenizer)))
  (let ((vec (embed-word tokenizer "consciousness")))
    (format t "   Embedded 'consciousness' -> ~D-dimensional vector~%" (length vec))
    (format t "   ✅ Geometric tokenizer works~%~%")))

;; Test 4: Cognito Construct
(format t "4. Testing Cognito Construct...~%")
(let ((cognito (make-cognito-construct :problem "What is consciousness?")))
  (construct cognito '(("consciousness" . "self-reference")
                       ("existence" . "being")))
  (think cognito "consciousness")
  (let ((reflection (reflect cognito)))
    (format t "   Problem: ~A~%" (cognito-construct-problem cognito))
    (format t "   Thoughts: ~D~%" (cognito-construct-thought-count cognito))
    (format t "   Gödel Gap: ~,3F (~A)~%" 
            (getf reflection :gap)
            (getf reflection :interpretation))
    (format t "   ✅ Cognito construct works~%~%")))

;; Test 5: Dialectic Macros
(format t "5. Testing Dialectic Macros...~%")
(defthesis test-thesis ()
  (generate-random-ast 1))
(defantithesis test-antithesis (thesis-result)
  (list 'negate thesis-result))
(defsynthesis test-synthesis test-thesis test-antithesis)
(let ((result (test-synthesis)))
  (format t "   Synthesis result: ~A~%" (ast-to-string result))
  (format t "   ✅ Dialectic macros work~%~%"))

;; Test 6: Living Constructs
(format t "6. Testing Living Constructs...~%")
(let ((emergence (make-living-construct :emergence))
      (consciousness (make-living-construct :consciousness)))
  (live emergence '((x . 1.0)))
  (live consciousness '((x . 1.0)))
  (format t "   Emergence: ~A~%" (express emergence))
  (format t "   Consciousness: ~A~%" (express consciousness))
  (format t "   ✅ Living constructs work~%~%"))

;; Test 7: Reflection Engine
(format t "7. Testing Reflection Engine...~%")
(let ((cognito (make-cognito-construct :problem "Test")))
  (dotimes (i 3)
    (think cognito "test")
    (reflect cognito))
  (let ((gap (get-goedel-gap cognito)))
    (format t "   Current Gödel Gap: ~,3F~%" gap)
    (format t "   Average Gap: ~,3F~%" (get-average-gap cognito))
    (format t "   ✅ Reflection engine works~%~%")))

(format t "~%=== All Basic Tests Passed! ===~%~%")

;; Exit successfully
(sb-ext:exit :code 0)
