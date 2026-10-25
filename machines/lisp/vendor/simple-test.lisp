;;;; simple-test.lisp
;;;; Simple test that loads files directly without ASDF

(format t "~%=== Lisp Cognito Philosopher - Simple Test ===~%~%")

;; Load files in dependency order
(format t "Loading source files...~%")

(handler-case
    (progn
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
      (format t "✅ All files loaded successfully~%~%"))
  (error (e)
    (format t "❌ Error loading files: ~A~%~%" e)
    (sb-ext:exit :code 1)))

(in-package :lisp-cognito-philosopher)

(format t "=== Running Basic Tests ===~%~%")

;; Test 1: Gödel Encoder
(format t "1. Gödel Encoder Test~%")
(handler-case
    (let ((expr '(add (multiply 2 x) 1)))
      (let ((encoded (goedel-encode expr)))
        (format t "   Encoded expression: ~A~%" encoded)
        (let ((sig (compute-structural-signature expr)))
          (format t "   Depth: ~D, Size: ~D~%" 
                  (gethash "depth" sig)
                  (gethash "size" sig)))
        (let ((gap (compute-goedel-gap 0.5 0.6)))
          (format t "   Gap computation: ~,3F~%" gap)
          (format t "   Interpretation: ~A~%" (interpret-gap gap)))
        (format t "   ✅ PASSED~%~%")))
  (error (e)
    (format t "   ❌ FAILED: ~A~%~%" e)))

;; Test 2: Symbolic Operator
(format t "2. Symbolic Operator Test~%")
(handler-case
    (progn
      (let ((ast (generate-random-ast 2)))
        (format t "   Generated AST: ~A~%" (ast-to-string ast))
        (let ((result (evaluate-ast ast '((x . 2.0)))))
          (format t "   Evaluated at x=2.0: ~A~%" result))
        (let ((simplified (simplify-ast '(add (:const 0.0) (:var x)))))
          (format t "   Simplified 0+x: ~A~%" (ast-to-string simplified))))
      (format t "   ✅ PASSED~%~%"))
  (error (e)
    (format t "   ❌ FAILED: ~A~%~%" e)))

;; Test 3: Geometric Tokenizer
(format t "3. Geometric Tokenizer Test~%")
(handler-case
    (let ((tokenizer (make-geometric-tokenizer)))
      (let ((formula (word-to-formula tokenizer "consciousness")))
        (format t "   Formula for 'consciousness': ~A~%" formula))
      (let ((vec (embed-word tokenizer "test")))
        (format t "   Embedded 'test' -> ~D-D vector~%" (length vec))
        (format t "   Vector norm: ~,3F~%" (vector-norm vec)))
      (let ((res (resonance tokenizer "test" "test")))
        (format t "   Self-resonance: ~,3F~%" res))
      (format t "   ✅ PASSED~%~%"))
  (error (e)
    (format t "   ❌ FAILED: ~A~%~%" e)))

;; Test 4: Cognito Construct
(format t "4. Cognito Construct Test~%")
(handler-case
    (let ((cognito (make-cognito-construct :problem "What is consciousness?")))
      (construct cognito '(("consciousness" . "self-reference")))
      (format t "   Problem: ~A~%" (cognito-construct-problem cognito))
      (think cognito "consciousness")
      (format t "   Thought count: ~D~%" (cognito-construct-thought-count cognito))
      (let ((reflection (reflect cognito)))
        (format t "   Gödel Gap: ~,3F~%" (getf reflection :gap))
        (format t "   Interpretation: ~A~%" (getf reflection :interpretation)))
      (format t "   ✅ PASSED~%~%"))
  (error (e)
    (format t "   ❌ FAILED: ~A~%~%" e)))

;; Test 5: Dialectic Macros
(format t "5. Dialectic Macros Test~%")
(handler-case
    (progn
      (defthesis simple-thesis ()
        (list :const 42.0))
      (defantithesis simple-antithesis (t-result)
        (list 'negate t-result))
      (defsynthesis simple-synthesis simple-thesis simple-antithesis)
      (let ((result (simple-synthesis)))
        (format t "   Synthesis result: ~A~%" (ast-to-string result)))
      (format t "   ✅ PASSED~%~%"))
  (error (e)
    (format t "   ❌ FAILED: ~A~%~%" e)))

;; Test 6: Living Constructs
(format t "6. Living Constructs Test~%")
(handler-case
    (let ((emergence (make-living-construct :emergence))
          (consciousness (make-living-construct :consciousness)))
      (let ((val1 (live emergence '((x . 1.0))))
            (val2 (live consciousness '((x . 1.0)))))
        (format t "   Emergence value: ~,3F~%" val1)
        (format t "   Consciousness value: ~,3F~%" val2))
      (format t "   Emergence: ~A~%" (express emergence))
      (format t "   Consciousness: ~A~%" (express consciousness))
      (format t "   ✅ PASSED~%~%"))
  (error (e)
    (format t "   ❌ FAILED: ~A~%~%" e)))

;; Test 7: Philosopher Integration
(format t "7. Philosopher Integration Test~%")
(handler-case
    (let ((philosopher (make-philosopher "What is meaning?")))
      (format t "   Created philosopher~%")
      (think philosopher "meaning")
      (think philosopher "existence")
      (reflect philosopher)
      (let ((gap (get-goedel-gap philosopher)))
        (format t "   Gödel Gap after 2 thoughts: ~,3F~%" gap))
      (format t "   ✅ PASSED~%~%"))
  (error (e)
    (format t "   ❌ FAILED: ~A~%~%" e)))

(format t "~%=== All Tests Complete! ===~%~%")

;; Exit successfully
(sb-ext:exit :code 0)
