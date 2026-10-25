;;;; bridge-protocol.lisp — Mini-JSON + Framing + Op-Dispatch der SBCL-Brücke.
;;;;
;;;; Protokoll (machines/lisp_bridge.py):
;;;;   * EINE S-Expression-Auftragzeile auf stdin, mit `read` geparst
;;;;     (cl-json unnötig — Vendor package.lisp nutzt nur (:use #:cl)).
;;;;   * Antwort: eine Zeile Mini-JSON zwischen ;;; RESULT-BEGIN und
;;;;     ;;; RESULT-END; alles andere auf stdout (Vendor-Logs, Philosopher-
;;;;     Text) wird von Python zwischen den Marken verworfen.
;;;;   * `(setf *log-level* :none)` (utils.lisp:137) killt die Emoji-Logs —
;;;;     should-log-p liefert NIL für jeden Level, sobald :none die aktuelle
;;;;     Ebene ist (position nil im Vergleicher).
;;;;   * ASCII-rein (Plan-Risiko 2: Bignum-Exponenten + SBCL-Locale); Unicode
;;;;     läuft Python-seitig.
;;;;   * Fehler erzeugen {"error": ...} im Frame und Exit-Code 0 — keine
;;;;     stillen False-Passes; Absturz des Prozesses → Python-Fehlerkanal.
;;;;
;;;; Ops (:op "..." :arg value ...):
;;;;   ping       — Lisp-Version + Primzahl-/Konstanteninventar (P1-Selbsttest)
;;;;   goedel     — :expr STRING → prin1-Kanon-Zeichenkette + goedel-encode
;;;;   gap        — :pred NUMBER|LIST, :actual NUMBER|LIST → compute-goedel-gap
;;;;   sh72       — :words LIST → SH-72-embeddings + Gematria + Resonanzen (P1)
;;;;   ast_eval   — :seed N, :count N, :xs LIST → LCG-ASTs + Auswertungen (P1)
;;;;   philosopher— :seed N, :cycles N, :queries LIST (:problem STRING) →
;;;;                Gap-History + Zähler (P2 strukturell: sxhash-Kanäle)
;;;;   simplify_ast— :ast NESTED-LIST → Vendor-Reduktion + Entlarvung
;;;;                (revealed = ('identity','x') bei triviale Identität →
;;;;                nacktes Var-Blatt); SP8 hermeneutischer Zirkel

(in-package #:lisp-cognito-philosopher)

;;; ============================================================================
;;; MINI-JSON
;;; ============================================================================

(defparameter *json-allow-nonascii* nil
  "NIL = \u00XX-Escape für >127 (Brücke ist ASCII-rein).")

(defun jescape (string)
  "JSON-Escape: \" \\ und Steuerzeichen; >127 als \\u00XX (Latin-1-Sicht)."
  (with-output-to-string (out)
    (loop for ch across string do
          (let ((code (char-code ch)))
            (cond
              ((char= ch #\")
               (write-string "\\\"" out))
              ((char= ch #\\)
               (write-string "\\\\" out))
              ((char= ch #\newline)
               (write-string "\\n" out))
              ((char= ch #\tab)
               (write-string "\\t" out))
              ((char= ch #\Return)
               (write-string "\\r" out))
              ((< code 32)
               (format out "\\u~4,'0X" code))
              ((and (>= code 128) (not *json-allow-nonascii*))
               (format out "\\u~4,'0X" code))
              (t
               (write-char ch out)))))))

(defun jnumber-string (n)
  "Integer ~D, sonst double-float ~,17E — aber den d-Exponentenmarker (SBCL)
   zu 'e' umgeschrieben, damit Python json.loads parst (E+000 ist legal)."
  (typecase n
    (integer (format nil "~D" n))
    (t (substitute-if #\e
                      (lambda (ch) (or (char= ch #\d) (char= ch #\D)))
                      (format nil "~,17E" (coerce n 'double-float))))))

(defun jwrite (value &optional (stream *standard-output*))
  "Mini-JSON-Writer: NIL → null, T → true, Keyword → String (lowercase),
   Vector/Cons → Array, Hash-Table → Objekt (sortierte Keys für Determinismus)."
  (typecase value
    (null (write-string "null" stream))
    ((eql t) (write-string "true" stream))
    (keyword
     (cond ((eq value :true) (write-string "true" stream))
           ((eq value :false) (write-string "false" stream))
           (t (format stream "\"~A\""
                      (jescape (string-downcase (symbol-name value)))))))
    (string (format stream "\"~A\"" (jescape value)))
    (number (write-string (jnumber-string value) stream))
    (symbol
     (write-string "\"" stream)
     (write-string (jescape (format nil "~A" value)) stream)
     (write-string "\"" stream))
    (vector
     (write-char #\[ stream)
     (loop for i from 0 below (length value)
           do (when (plusp i) (write-char #\, stream))
              (jwrite (aref value i) stream))
     (write-char #\] stream))
    (hash-table
     (write-char #\{ stream)
     (let ((keys (sort (copy-seq (loop for key being the hash-keys of value
                                       collect key)) #'string<)))
       (loop for key in keys
             for i from 0 do
             (when (plusp i) (write-char #\, stream))
             (jwrite (format nil "~A" key) stream)
             (write-char #\: stream)
             (jwrite (gethash key value) stream)))
     (write-char #\} stream))
    (cons
     (if (consp (car value))              ; Alist ((k . v) ...) → Objekt
         (jwrite-alist value stream)
       (let ((len (length value)))        ; echte Liste → Array (Zahlen, Strings)
         (write-char #\[ stream)
         (loop for i from 0 below len
               do (when (plusp i) (write-char #\, stream))
                  (jwrite (nth i value) stream))
         (write-char #\] stream))))
    (t
     (jwrite (format nil "~A" value) stream))))

(defun jemit-line (value &optional (stream *standard-output*))
  "Ein JSON-Dokument in einer Zeile."
  (jwrite value stream)
  (terpri stream))

;;; ============================================================================
;;; PROTOCOL FRAME
;;; ============================================================================

(defparameter *result-marker-begin* ";;; RESULT-BEGIN")
(defparameter *result-marker-end* ";;; RESULT-END")

(defun jwrite-alist (alist stream)
  "Objekt-Writer für die Op-Antworten ((key . value)-Conses); Keys als
   Strings, sortiert nicht nötig (Definitionsjenseits der Fixture-Ordnung)."
  (write-char #\{ stream)
  (loop for entry in alist
        for i from 0 do
        (when (plusp i) (write-char #\, stream))
        (jwrite (format nil "~A" (car entry)) stream)
        (write-char #\: stream)
        (jwrite (cdr entry) stream))
  (write-char #\} stream))

(defun emit-result-frame (object &optional (stream *standard-output*))
  "Antwort-Rahmen: Marke, JSON-Objekt-Zeile, Marke, Flush."
  (format stream "~&~A~%" *result-marker-begin*)
  (jwrite-alist object stream)
  (terpri stream)
  (format stream "~A~%" *result-marker-end*)
  (finish-output stream))

;;; ============================================================================
;;; OP HANDLERS (P1 = bitgenau testbar gegen machines/lisp_parity.py)
;;; ============================================================================

(defun to-double-vector (list)
  "Python-Liste → simple double-float-Vector (gap/embedding-Semantik)."
  (coerce (mapcar (lambda (x) (coerce x 'double-float)) list) 'vector))

(defun number-or-vector (value)
  "NUMBER bleibt, LIST (leer erlaubt) → double-Vector."
  (typecase value
    (number value)
    (list (to-double-vector value))
    (t value)))

(defun handle-ping (&key &allow-other-keys)
  (list (cons "implementation" (lisp-implementation-version))
        (cons "package" "lisp-cognito-philosopher")
        (cons "primes" (coerce *primes* 'vector))
        (cons "unary" (coerce *unary-operators* 'vector))
        (cons "binary" (coerce *binary-operators* 'vector))
        (cons "constants" (coerce *constants* 'vector))
        (cons "rng_state_default" *rng-state*)))

(defun handle-goedel (&key expr &allow-other-keys)
  "expr: STRING oder S-Expression (vom `read` geparst) — encode beides als
   prin1-to-string-Kanon (Strings bekommen Anführungszeichen!)."
  (let* ((canonical (prin1-to-string expr))
         (number (goedel-encode-string canonical)))
    (list (cons "expr" expr)
          (cons "canonical" canonical)
          (cons "goedel_number" number)
          (cons "chars" (length canonical)))))

(defun handle-gap (&key pred actual &allow-other-keys)
  "Lisp-Gap-Kanon (goedel-encoder.lisp:87-150): Nenner ist |actual| mit
   1e-9-Guard — VERSCHIEDEN von uni_241s ‖state‖+1e-6 (Python apply_gap)!"
  (let* ((p (number-or-vector pred))
         (a (number-or-vector actual))
         (gap (compute-goedel-gap p a)))
    (list (cons "gap" gap)
          (cons "interpretation" (interpret-gap gap))
          (cons "kind" (typecase a
                         (vector "vector")
                         (number "scalar")
                         (t "string"))))))

(defun handle-sh72 (&key words &allow-other-keys)
  "SH-72-Fallback (leerer Lexikon): embed + Gematria + Resonanzen i<j."
  (let* ((tokenizer (make-geometric-tokenizer))
         (names (mapcar (lambda (w) (format nil "~A" w)) words))
         (embeds (mapcar (lambda (w) (embed-word tokenizer w)) names))
         (resonances
           (loop for w1 in names
                 for i from 0
                 append (loop for w2 in names
                              for j from 0
                              while (< j i)
                              collect (list (cons "i" i)
                                            (cons "j" j)
                                            (cons "r" (resonance tokenizer w1 w2)))))))
    (list (cons "words" names)
          (cons "gematria" (mapcar (lambda (w) (get-gematria w)) names))
          (cons "embed" embeds)                       ; Liste von Vectors → Array
          ;; VECTOR → Array-Regel: Objekt-Paare als Array-of-Objects.
          (cons "resonances" (coerce resonances 'vector)))))

(defun handle-ast-eval (&key (seed 123456789) (count 3) (xs (list 1.0d0))
                          &allow-other-keys)
  "LCG-gepunktete AST-Generation (symbolic-operator.lisp:56-79 nutzt
   deterministic-random!) + Auswertung an allen x; Fehler je AST als null."
  (setf *rng-state* (logand seed #xFFFFFFFF))
  (let ((results ()))
    (dotimes (i count)
      (let ((ast (generate-random-ast 0 3)))
        (push (list (cons "index" i)
                    (cons "sexpr" (prin1-to-string ast))
                    (cons "rendered" (ast-to-string ast))
                    (cons "size" (ast-size ast))
                    (cons "depth" (ast-depth ast))
                    (cons "values"
                          (loop for x in (mapcar (lambda (x) (coerce x 'double-float)) xs)
                                collect (handler-case
                                            (evaluate-ast ast (list (cons 'x x)))
                                          (error (e)
                                            (list (cons "error" (format nil "~A" e))))))))
              results)))
    (list (cons "seed" seed)
          (cons "xs" (coerce (mapcar (lambda (x) (coerce x 'double-float)) xs) 'vector))
          ;; VECTOR → Array-Regel: Liste von Objekt-Alists als Vector markieren.
          (cons "asts" (coerce (reverse results) 'vector)))))

(defun handle-philosopher (&key (seed 123456789) (problem "self-reference")
                             (n-cycles 2)
                             (queries (list "consciousness" "existence" "meaning"))
                             (self-modify-threshold 0.5)
                             &allow-other-keys)
  "P2 strukturell: sxhash-/Text-Kanäle sind nicht bitportabel; die
   Struktur (History-Länge, Zähler, Interpretations-Keys) ist fix."
  (setf *rng-state* (logand seed #xFFFFFFFF))
  (let* ((cognito (make-philosopher problem))
         (report (run-philosopher cognito
                                  :queries queries
                                  :n-cycles n-cycles
                                  :self-modify-threshold self-modify-threshold
                                  :generate-text nil))
         (history (cognito-construct-goedel-gap-history cognito)))
    (declare (ignore report))
    (list (cons "problem" problem)
          (cons "n_cycles" n-cycles)
          (cons "queries" (mapcar (lambda (q) (format nil "~A" q)) queries))
          (cons "gap_history" (coerce history 'vector))
          (cons "history_length" (length history))
          (cons "thought_count" (cognito-construct-thought-count cognito))
          (cons "state" (cognito-construct-state cognito))
          (cons "self_model" (cognito-construct-self-model cognito)))))

;;; ============================================================================
;;; SIMPLIFY_AST — SP8 hermeneutischer Zirkel (Python -> Lisp -> Python)
;;; ============================================================================

(defun normalize-python-ast (ast)
  "Python-Form (verschachtelte Listen, Op-Strings, 'x', bare floats) in die
   Vendor-Form spiegeln (symbolic-operator.lisp:118-182): 'x' -> (:var X),
   Zahl -> (:const d), Op-String -> interned Symbol; Liste -> rekursiv."
  (cond
    ((numberp ast) (list :const (coerce ast 'double-float)))
    ((and (stringp ast) (string= ast "x")) (list :var 'x))
    ((stringp ast) (intern (string-upcase ast) :lisp-cognito-philosopher))
    (t (cons (normalize-python-ast (first ast))
             (mapcar #'normalize-python-ast (rest ast))))))

(defun dewrap-python-ast (ast)
  "Vendor-Form zurueck in die Python-Form: :const -> bare Zahl, :var -> \"x\",
   Op-Symbol -> lowercase String, Liste -> Liste (jwrite: Op-String zuerst,
   Car niemals Cons -> ohne Alist-Mehrdeutigkeit)."
  (cond
    ((numberp ast) ast)
    ((atom ast) (string-downcase (symbol-name ast)))
    ((eq (first ast) :const) (second ast))
    ((eq (first ast) :var) "x")
    (t (cons (string-downcase (symbol-name (first ast)))
             (mapcar #'dewrap-python-ast (rest ast))))))

(defun trivial-identity-node-p (ast)
  "Triviale Identitaet als BINARY-Knoten — add(0,x) / add(x,0) /
   multiply(1,x) / multiply(x,1): mathematisch x, aber KEIN identity-Knoten,
   von compile_rule (composite.py:76-88: nur ('identity','x')) unsichtbar."
  (and (consp ast)
       (eql (length ast) 3)
       (member (first ast) '(add multiply))
       (or (and (eq (first ast) 'add)
                (or (const-zerop (second ast)) (const-zerop (third ast))))
           (and (eq (first ast) 'multiply)
                (or (const-onep (second ast)) (const-onep (third ast)))))))

(defun const-zerop (x)
  (and (consp x) (eq (first x) :const) (zerop (second x))))

(defun const-onep (x)
  (and (consp x) (eq (first x) :const) (= 1.0d0 (second x))))

(defun contains-trivial-identity-p (ast)
  (cond ((not (consp ast)) nil)
        ((member (first ast) '(:const :var)) nil)
        (t (or (trivial-identity-node-p ast)
               (some #'contains-trivial-identity-p (rest ast))))))

(defun contains-identity-wrapper-p (ast)
  (cond ((not (consp ast)) nil)
        ((member (first ast) '(:const :var)) nil)
        (t (or (and (eq (first ast) 'identity) (eql (length ast) 2))
               (some #'contains-identity-wrapper-p (rest ast))))))

(defun count-ast-nodes (ast)
  (cond ((atom ast) 1)
        (t (1+ (reduce #'+ (mapcar #'count-ast-nodes (rest ast)))))))

(defun handle-simplify-ast (&key ast &allow-other-keys)
  "SP8 — Reflexionsphase des Zirkels: den Python-AST durch den Vendor-Reduktor
   schicken und DANACH entlarven. Der Vendor wascht den identity-Wrapper weg
   (apply-simplification-rules: (identity (first args)) -> (first args)); die
   Entlarvungs-Schicht entscheidet darueber, ob die Reduktion zur Tautologie
   ruehrte: landet der AST auf dem nackten Var-Blatt x UND enthielt die
   Ursprungsform eine triviale Identitaet oder einen identity-Wrapper, wird
   das Resultat als ('identity','x') ausgewiesen — der Knoten, den
   compile_rule als Selbst-Transition uebersetzt (LOOP + expand_basis)."
  (unless (or (listp ast) (stringp ast) (numberp ast))
    (error "simplify_ast: :ast fehlt (Liste, String-Blatt oder Zahl erwartet)"))
  (let* ((norm (normalize-python-ast ast))
         (flagged (or (contains-trivial-identity-p norm)
                      (contains-identity-wrapper-p norm)))
         (simp (simplify-ast norm))
         (leaf-var (and (consp simp) (eq (first simp) :var)))
         (revealed (and flagged leaf-var)))
    (list (cons "ast_in" ast)          ; Raw-Form ist bereits Python-Form
          (cons "ast_simplified" (dewrap-python-ast simp))
          (cons "revealed" (if revealed :true :false))
          (cons "contains_trivial_identity" (if flagged :true :false))
          (cons "size_in" (count-ast-nodes ast))
          (cons "size_out" (count-ast-nodes simp)))))

(defparameter *bridge-ops*
  (list (cons "ping" #'handle-ping)
        (cons "goedel" #'handle-goedel)
        (cons "gap" #'handle-gap)
        (cons "sh72" #'handle-sh72)
        (cons "ast_eval" #'handle-ast-eval)
        (cons "philosopher" #'handle-philosopher)
        (cons "simplify_ast" #'handle-simplify-ast)))

(defparameter *bridge-aliases*
  '(("embed" . "sh72") ("resonance" . "sh72") ("ast" . "ast_eval")))

(defun bridge-dispatch (request)
  "(:op \"NAME\" :arg value ...) → Pliste der Op-Antwort; fehlendes :op → ping."
  (let* ((raw-op (getf request :op))
         (name (if raw-op (string-downcase (string raw-op)) "ping"))
         (key (or (cdr (assoc name *bridge-aliases* :test #'string=)) name)))
    (let ((handler (cdr (assoc key *bridge-ops* :test #'string=))))
      (unless handler
        (error "unknown bridge op: ~A" (or raw-op "ping")))
      (apply handler request))))

;;; ============================================================================
;;; MAIN
;;; ============================================================================

(defun run-bridge ()
  "Einen Auftrag aus stdin lesen, ausführen, Frame schreiben. Fehler →
   {\"error\": ...} im Frame, Exit 0."
  (handler-case
      (let* ((line (read-line *standard-input* nil nil))
             (request (when (and line (plusp (length (string-trim
                                                      '(#\space #\tab #\Return
                                                        #\Newline) line))))
                        (with-input-from-string (in line) (read in)))))
        (when (null request)
          (error "empty request (no stdin line)"))
        (emit-result-frame (bridge-dispatch request)))
    (error (e)
      (emit-result-frame
       (list (cons "error" (format nil "~A" e)))))))

(defun run-bridge-main ()
  ;; Loglevel nach dem Vendor-Load totlegen (utils.lisp-Emoji-Logs).
  (setf *log-level* :none)
  (run-bridge)
  (finish-output *standard-output*)
  (finish-output *standard-error*))