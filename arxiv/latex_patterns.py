# ==================================================
# 1. COMMANDS - NO ARGUMENT
# ==================================================
# Commands of the form \command. Remove completely.

REMOVE_COMMANDS = [

    # --- Text alignment ---
    r'\centering',
    r'\raggedright',
    r'\raggedleft',

    # --- Font size ---
    r'\tiny',
    r'\scriptsize',
    r'\footnotesize',
    r'\small',
    r'\normalsize',
    r'\large',
    r'\Large',
    r'\LARGE',
    r'\huge',
    r'\Huge',

    # --- Font family / style ---
    r'\normalfont',
    r'\rmfamily',
    r'\sffamily',
    r'\ttfamily',
    r'\mdseries',
    r'\bfseries',
    r'\itshape',
    r'\slshape',
    r'\scshape',

    # --- Paragraph indentation ---
    r'\noindent',
    r'\indent',

    # --- Line and page breaks ---
    r'\newline',
    r'\\',
    r'\linebreak',
    r'\nolinebreak',
    r'\pagebreak',
    r'\nopagebreak',
    r'\newpage',
    r'\clearpage',
    r'\cleardoublepage',

    # --- Vertical / horizontal spacing ---
    r'\smallskip',
    r'\medskip',
    r'\bigskip',
    r'\hfill',

    # --- Paragraph command ---
    r'\par',

]


# ==================================================
# 2. COMMANDS - PRESERVE ARGUMENT
# ==================================================
# Commands of the form \command{arg}. Remove command 
# but preserve wrapped content.

PRESERVE_CONTENT = [

    # --- Text formatting ---
    r'\textbf',
    r'\textit',
    r'\texttt',
    r'\textrm',
    r'\textsf',
    r'\emph',
    r'\underline',

    # --- Text containers ---

r'\text',
    r'\mbox',
    r'\textnormal',

]


# ==================================================
# 3. COMMANDS - REMOVE ARGUMENT
# ==================================================
# Commands of the form \command[arg1]{arg2}. Remove 
# command and (optional) arguments.

REMOVE_CONTENT = [

    # --- Graphics ---
    r'\includegraphics',

    # --- Labels / document references ---
    r'\label',

    # --- Bibliography ---
    r'\bibliography',
    r'\bibliographystyle',

    # --- Formatting ---
    r'\thispagestyle',
    r'\vspace',

]


# ==================================================
# 4. ENVIRONMENTS
# ==================================================
# Environments of the form \begin{env} ... \end{env}.
# Remove completely.

REMOVE_ENVIRONMENTS = [

    # --- Figures ---
    'figure',
    'figure*',

    # --- Tables ---
    'table',
    'table*',
    'tabular',
    'tabular*',

]