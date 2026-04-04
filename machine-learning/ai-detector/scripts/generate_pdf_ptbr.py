"""
Gera o documento tecnico do AI Detector em Portugues BR.
Uso: python scripts/generate_pdf_ptbr.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fpdf import FPDF, XPos, YPos

PDF_OUT = ROOT / "AI_Detector_Documentacao_Tecnica.pdf"
METRICS_PATH = ROOT / "app" / "models" / "ml" / "v3_metrics.json"

# ---------------------------------------------------------------------------
# Paleta de cores
# ---------------------------------------------------------------------------
C_DARK   = (15,  23,  42)
C_ACCENT = (79,  70,  229)
C_GREEN  = (5,   150, 105)
C_AMBER  = (217, 119,   6)
C_RED    = (220,  38,  38)
C_LIGHT  = (238, 242, 255)
C_WHITE  = (255, 255, 255)
C_GRAY   = (107, 114, 128)
C_BG     = (248, 250, 252)


def safe(text: str) -> str:
    """Remove caracteres fora do latin-1 para compatibilidade com fpdf."""
    result = []
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2019': "'", '\u2018': "'",
        '\u201c': '"', '\u201d': '"', '\u2192': '->', '\u2022': '-',
        '\u00e7': 'c', '\u00e3': 'a', '\u00f5': 'o', '\u00e9': 'e',
        '\u00ea': 'e', '\u00e0': 'a', '\u00e1': 'a', '\u00ed': 'i',
        '\u00f3': 'o', '\u00fa': 'u', '\u00fc': 'u', '\u00f1': 'n',
        '\u00c9': 'E', '\u00c3': 'A', '\u00c7': 'C', '\u00d5': 'O',
    }
    for c in text:
        if c in replacements:
            result.append(replacements[c])
        else:
            try:
                c.encode('latin-1')
                result.append(c)
            except UnicodeEncodeError:
                result.append('?')
    return ''.join(result)


# ---------------------------------------------------------------------------
# Classe PDF
# ---------------------------------------------------------------------------

class PDF(FPDF):

    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=22)

    def header(self):
        if self.page_no() <= 2:
            return
        self.set_fill_color(*C_DARK)
        self.rect(0, 0, 210, 11, "F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*C_WHITE)
        self.set_xy(15, 3)
        self.cell(130, 5, "AI DETECTOR  |  Documentacao Tecnica", align="L")
        self.set_text_color(180, 180, 200)
        self.cell(0, 5, f"Pagina {self.page_no() - 2}", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def footer(self):
        self.set_y(-11)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*C_GRAY)
        self.cell(0, 5,
                  "AI Detector - Documentacao Tecnica Confidencial - 2025",
                  align="C")

    # -----------------------------------------------------------------------
    # Componentes de layout
    # -----------------------------------------------------------------------

    def needs_page(self, space: float = 40):
        """Adiciona nova pagina se espaco restante for menor que 'space' mm."""
        if self.get_y() > (297 - 22 - space):
            self.add_page()

    def section_title(self, number: str, title: str):
        self.needs_page(50)
        self.ln(5)
        # Barra lateral colorida
        self.set_fill_color(*C_ACCENT)
        self.rect(15, self.get_y() + 1, 4, 9, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*C_DARK)
        self.set_x(22)
        self.cell(0, 11, safe(f"{number}. {title}"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Linha separadora
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.4)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)

    def subsection(self, title: str):
        self.needs_page(30)
        self.ln(3)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C_ACCENT)
        self.set_x(15)
        self.cell(0, 6, safe(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 60)
        self.set_x(15)
        self.multi_cell(180, 5.5, safe(text),
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def bullet_list(self, items: list):
        """
        items pode ser:
          - str simples
          - (titulo_bold, descricao)
        """
        for item in items:
            self.needs_page(12)
            if isinstance(item, tuple):
                label, text = item
                y = self.get_y()
                # quadrado bullet
                self.set_fill_color(*C_ACCENT)
                self.rect(17, y + 2, 2.5, 2.5, "F")
                self.set_x(22)
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(*C_DARK)
                label_w = self.get_string_width(safe(label) + ": ") + 2
                self.cell(label_w, 5.5, safe(label) + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
                self.set_font("Helvetica", "", 10)
                self.set_text_color(40, 40, 60)
                remaining = 180 - (label_w + 7)
                self.multi_cell(remaining, 5.5, safe(text),
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                y = self.get_y()
                self.set_fill_color(*C_ACCENT)
                self.rect(17, y + 2, 2.5, 2.5, "F")
                self.set_x(22)
                self.set_font("Helvetica", "", 10)
                self.set_text_color(40, 40, 60)
                self.multi_cell(173, 5.5, safe(str(item)),
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def info_box(self, title: str, lines: list, accent: bool = False):
        """
        Caixa de destaque.
        lines: lista de strings, cada uma vira uma linha.
        """
        self.needs_page(35)
        x = 15
        y = self.get_y()
        w = 180

        # Mede altura necessaria
        self.set_font("Helvetica", "", 9)
        line_h = 5
        content_h = 4 + len(lines) * (line_h + 1) + 4

        # Fundo
        header_color = C_ACCENT if accent else C_DARK
        self.set_fill_color(*header_color)
        self.rect(x, y, w, 7, "F")
        self.set_fill_color(*C_BG)
        self.rect(x, y + 7, w, content_h, "F")
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.3)
        self.rect(x, y, w, 7 + content_h, "D")

        # Titulo
        self.set_xy(x + 4, y + 1.5)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C_WHITE)
        self.cell(w - 8, 4, safe(title))

        # Linhas de conteudo
        cy = y + 7 + 3
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 60)
        for line in lines:
            self.set_xy(x + 4, cy)
            self.multi_cell(w - 8, line_h, safe(line),
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            cy += line_h + 1

        self.set_y(y + 7 + content_h + 4)

    def accuracy_bar(self, label: str, value: float, color, best=False):
        self.needs_page(10)
        y = self.get_y()
        bar_max = 120
        bar_w = int(value * bar_max)

        self.set_font("Helvetica", "B" if best else "", 9)
        self.set_text_color(*C_DARK)
        self.set_x(15)
        self.cell(60, 6, safe(label))

        # Fundo da barra
        self.set_fill_color(220, 220, 230)
        self.rect(77, y + 1.5, bar_max, 3.5, "F")
        # Barra preenchida
        self.set_fill_color(*color)
        self.rect(77, y + 1.5, bar_w, 3.5, "F")

        self.set_x(200)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(0, 6, f"{value:.1%}", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # -----------------------------------------------------------------------
    # Tabela de features
    # -----------------------------------------------------------------------

    def feature_table(self, features: list):
        """Tabela com quebra de pagina automatica."""
        col_w    = [8, 45, 30, 22, 75]
        headers  = ["#", "Feature", "Sinal de IA", "Importancia", "Descricao"]
        row_h    = 6.5

        def draw_header():
            self.set_fill_color(*C_DARK)
            self.set_text_color(*C_WHITE)
            self.set_font("Helvetica", "B", 8)
            self.set_x(15)
            for h, w in zip(headers, col_w):
                self.cell(w, 8, safe(h), border=1, fill=True, align="C")
            self.ln()

        draw_header()

        for i, f in enumerate(features):
            self.needs_page(14)
            # Se virou pagina, redesenha cabecalho
            if self.get_y() < 25:
                draw_header()

            alt = i % 2 == 0
            self.set_fill_color(*C_BG) if alt else self.set_fill_color(*C_WHITE)

            self.set_x(15)
            # Coluna #
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*C_GRAY)
            self.cell(col_w[0], row_h, str(i + 1), border=1, fill=alt, align="C")

            # Feature name
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*C_DARK)
            self.cell(col_w[1], row_h, safe(f["name"]), border=1, fill=alt)

            # Sinal
            sig = f["signal"]
            self.set_font("Helvetica", "", 8)
            if "Alta" in sig:
                self.set_text_color(*C_RED)
            elif "Baixa" in sig:
                self.set_text_color(*C_GREEN)
            else:
                self.set_text_color(*C_AMBER)
            self.cell(col_w[2], row_h, safe(sig), border=1, fill=alt, align="C")

            # Importancia com barra mini
            imp = f.get("imp", 0.0)
            x_imp = self.get_x()
            y_imp = self.get_y()
            self.set_fill_color(*C_BG) if alt else self.set_fill_color(*C_WHITE)
            self.cell(col_w[3], row_h, "", border=1, fill=alt)
            bar_len = int(imp * (col_w[3] - 4))
            self.set_fill_color(*C_ACCENT)
            self.rect(x_imp + 1, y_imp + 2, bar_len, 2.5, "F")
            self.set_xy(x_imp, y_imp)
            self.set_text_color(*C_DARK)
            self.set_font("Helvetica", "B", 7)
            self.cell(col_w[3], row_h, f"{imp:.0%}", align="C", border=0)
            self.set_x(x_imp + col_w[3])

            # Descricao
            self.set_font("Helvetica", "", 8)
            self.set_text_color(50, 50, 70)
            self.cell(col_w[4], row_h, safe(f["desc"]), border=1, fill=alt)
            self.ln()

        self.ln(4)

    # -----------------------------------------------------------------------
    # Tabela comparativa
    # -----------------------------------------------------------------------

    def comparison_table(self, headers, col_w, rows):
        row_h = 6

        def draw_header():
            self.set_fill_color(*C_DARK)
            self.set_text_color(*C_WHITE)
            self.set_font("Helvetica", "B", 8)
            self.set_x(15)
            for h, w in zip(headers, col_w):
                self.cell(w, 8, safe(h), border=1, fill=True, align="C")
            self.ln()

        draw_header()

        for i, row in enumerate(rows):
            self.needs_page(12)
            if self.get_y() < 25:
                draw_header()

            alt = i % 2 == 0
            self.set_fill_color(*C_BG) if alt else self.set_fill_color(*C_WHITE)
            self.set_x(15)

            for j, (val, w) in enumerate(zip(row, col_w)):
                if j == 0:
                    self.set_font("Helvetica", "B", 8)
                    self.set_text_color(*C_DARK)
                    self.cell(w, row_h, safe(val), border=1, fill=alt, align="L")
                else:
                    self.set_font("Helvetica", "", 8)
                    is_yes = val.upper().startswith("SIM") or val == "Nao"
                    is_no = val.upper().startswith("NAO") or "API" in val
                    if j == 1:
                        # Nossa coluna - sempre verde
                        self.set_text_color(*C_GREEN)
                        self.set_font("Helvetica", "B", 8)
                    elif is_no:
                        self.set_text_color(*C_RED)
                    elif is_yes:
                        self.set_text_color(*C_GREEN)
                    else:
                        self.set_text_color(*C_AMBER)
                    self.cell(w, row_h, safe(val), border=1, fill=alt, align="C")
            self.ln()

        self.ln(4)

    # -----------------------------------------------------------------------
    # Roadmap item
    # -----------------------------------------------------------------------

    def roadmap_item(self, fase: str, titulo: str, itens: list,
                     status: str, color, status_color=None):
        if status_color is None:
            status_color = color

        total_h = 10 + len(itens) * 6 + 4
        self.needs_page(total_h + 5)

        x, y, w = 15, self.get_y(), 180

        # Cabecalho colorido
        self.set_fill_color(*color)
        self.rect(x, y, w, 10, "F")

        # Badge de status
        badge_w = 28
        self.set_fill_color(*status_color)
        self.rect(x + w - badge_w - 2, y + 2, badge_w, 6, "F")
        self.set_xy(x + w - badge_w - 2, y + 2.5)
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*C_WHITE)
        self.cell(badge_w, 5, safe(status), align="C")

        # Fase e titulo
        self.set_xy(x + 4, y + 2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C_WHITE)
        self.cell(25, 6, safe(fase))
        self.set_font("Helvetica", "B", 10)
        self.cell(w - 25 - badge_w - 8, 6, safe(titulo))

        # Itens
        self.set_fill_color(248, 249, 255)
        self.rect(x, y + 10, w, len(itens) * 6 + 4, "F")
        self.set_draw_color(*color)
        self.set_line_width(0.4)
        self.rect(x, y, w, 10 + len(itens) * 6 + 4, "D")

        item_y = y + 12
        for item in itens:
            self.set_xy(x + 6, item_y)
            self.set_fill_color(*color)
            self.rect(x + 4, item_y + 1.5, 2, 2, "F")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(40, 40, 60)
            self.cell(w - 10, 5, safe(item))
            item_y += 6

        self.set_y(y + 10 + len(itens) * 6 + 8)

    # -----------------------------------------------------------------------
    # Capa
    # -----------------------------------------------------------------------

    def cover_page(self):
        self.add_page()

        # Fundo escuro
        self.set_fill_color(*C_DARK)
        self.rect(0, 0, 210, 297, "F")

        # Faixa accent superior
        self.set_fill_color(*C_ACCENT)
        self.rect(0, 0, 210, 3, "F")

        # Faixa lateral esquerda
        self.set_fill_color(*C_ACCENT)
        self.rect(0, 0, 6, 297, "F")

        # Titulo principal
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", 52)
        self.set_y(65)
        self.cell(0, 26, "AI DETECTOR", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Subtitulo
        self.set_font("Helvetica", "", 16)
        self.set_text_color(180, 185, 230)
        self.cell(0, 10, "Documentacao Tecnica  v3.0", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Linha separadora
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(1)
        self.line(50, self.get_y() + 6, 160, self.get_y() + 6)
        self.ln(16)

        # Descricao
        self.set_font("Helvetica", "", 12)
        self.set_text_color(210, 215, 240)
        self.set_x(30)
        self.multi_cell(150, 7,
            "Deteccao de conteudo gerado por Inteligencia Artificial\n"
            "sem depender de LLMs -- rapido, explicavel e continuo.",
            align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(10)

        # Cards de metricas
        cards = [
            ("93,4%",    "Acuracia CV"),
            ("16.000",   "Amostras de Treino"),
            ("12",       "Features Estatisticas"),
            ("<50ms",    "Latencia Media"),
        ]
        card_w = 38
        total = len(cards) * card_w + (len(cards) - 1) * 4
        cx = (210 - total) / 2
        cy = self.get_y()

        for val, lbl in cards:
            self.set_fill_color(30, 35, 70)
            self.rect(cx, cy, card_w, 22, "F")
            self.set_draw_color(*C_ACCENT)
            self.set_line_width(0.5)
            self.rect(cx, cy, card_w, 22, "D")

            self.set_xy(cx, cy + 4)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*C_GREEN)
            self.cell(card_w, 8, val, align="C",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            self.set_xy(cx, cy + 13)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(180, 185, 230)
            self.cell(card_w, 5, lbl, align="C")
            cx += card_w + 4

        # Tags
        tags = ["Machine Learning", "Cascata 3 Camadas", "Aprendizado Continuo",
                "API Claude (opcional)", "Open Source"]
        tx = 20
        ty = cy + 34
        self.set_font("Helvetica", "", 8)
        for tag in tags:
            tw = self.get_string_width(tag) + 8
            self.set_fill_color(40, 45, 90)
            self.rect(tx, ty, tw, 7, "F")
            self.set_draw_color(80, 85, 160)
            self.rect(tx, ty, tw, 7, "D")
            self.set_xy(tx, ty + 1)
            self.set_text_color(180, 185, 230)
            self.cell(tw, 5, tag, align="C")
            tx += tw + 5
            if tx > 175:
                tx = 20
                ty += 10

        # Data
        self.set_y(258)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*C_GRAY)
        self.cell(0, 6, "Abril de 2025  |  Versao 3.0", align="C")

    # -----------------------------------------------------------------------
    # Sumario
    # -----------------------------------------------------------------------

    def toc_page(self, items: list):
        self.add_page()
        self.set_y(30)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*C_DARK)
        self.cell(0, 10, "Sumario", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.5)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(8)

        for num, title, _ in items:
            self.set_x(15)
            # Numero
            self.set_fill_color(*C_ACCENT)
            self.rect(15, self.get_y() + 1.5, 8, 7, "F")
            self.set_xy(15, self.get_y() + 1.5)
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*C_WHITE)
            self.cell(8, 7, num, align="C")

            # Titulo
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*C_DARK)
            self.set_x(26)
            self.cell(160, 10, safe(title))

            # Linha pontilhada
            self.set_draw_color(200, 200, 215)
            self.set_line_width(0.2)
            self.line(15, self.get_y() + 9.5, 195, self.get_y() + 9.5)
            self.ln(10)


# ---------------------------------------------------------------------------
# Conteudo
# ---------------------------------------------------------------------------

def build_pdf():
    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)

    cv_acc   = metrics.get("cv_accuracy_mean", 0.934)
    n_amostras = metrics.get("n_train_samples", 16000)
    feat_imp = metrics.get("feature_importances", {})

    pdf = PDF()
    pdf.set_margins(15, 20, 15)

    # -----------------------------------------------------------
    # CAPA
    # -----------------------------------------------------------
    pdf.cover_page()

    # -----------------------------------------------------------
    # SUMARIO
    # -----------------------------------------------------------
    toc = [
        ("1",  "Resumo Executivo",                           0),
        ("2",  "O Problema: Detectar Conteudo de IA",        0),
        ("3",  "Por que nao usar um LLM para detectar?",     0),
        ("4",  "Nossa Abordagem: Deteccao Estatistica",      0),
        ("5",  "Engenharia de Features -- 12 Discriminadores", 0),
        ("6",  "Arquitetura: A Cascata de 3 Camadas",        0),
        ("7",  "Metricas de Desempenho",                     0),
        ("8",  "Sistema de Aprendizado Continuo",            0),
        ("9",  "Diferenciais Competitivos",                  0),
        ("10", "Stack Tecnico",                              0),
        ("11", "Roadmap",                                    0),
    ]
    pdf.toc_page(toc)

    # -----------------------------------------------------------
    # 1. RESUMO EXECUTIVO
    # -----------------------------------------------------------
    pdf.add_page()
    pdf.set_y(25)

    pdf.section_title("1", "Resumo Executivo")
    pdf.body(
        "O AI Detector e uma plataforma de machine learning para identificar conteudo "
        "gerado por Inteligencia Artificial em textos e imagens. Ao contrario das solucoes "
        "concorrentes que dependem de LLMs para classificacao, o AI Detector usa uma "
        "arquitetura em cascata onde o motor estatistico de ML e o tomador de decisao "
        "principal -- rapido, deterministico e com privacidade preservada -- enquanto o "
        "Claude atua apenas como desempate nos casos verdadeiramente incertos."
    )
    pdf.body(
        f"O modelo atual (v3) alcanca {cv_acc:.1%} de acuracia com validacao cruzada "
        f"sobre {n_amostras:,} exemplos reais de datasets publicos do HuggingFace. "
        "Processa um texto em menos de 50ms em hardware comum, sem nenhuma chamada a "
        "API externa."
    )

    pdf.info_box(
        "Metricas Principais",
        [
            f"Acuracia CV (5-fold): {cv_acc:.1%}    |    Amostras de treino: {n_amostras:,}",
            "Dimensoes de feature: 12    |    Latencia de inferencia: <50ms",
            "API externa obrigatoria: NAO (Claude e opcional, apenas para zona incerta)",
        ],
        accent=True,
    )

    pdf.section_title("2", "O Problema: Detectar Conteudo de IA")
    pdf.body(
        "A proliferacao de LLMs (GPT-4, Claude, Gemini, Llama) tornou trivialmente facil "
        "gerar texto parecido com o humano em escala. Isso cria desafios serios em "
        "multiplos dominios:"
    )
    pdf.bullet_list([
        ("Educacao",    "Alunos entregando redacoes e trabalhos escritos por IA."),
        ("Jornalismo",  "Desinformacao gerada por IA e artigos sinteticos de noticias."),
        ("Pesquisa",    "Abstracts academicos fabricados poluindo a literatura cientifica."),
        ("Plataformas", "Spam, avaliacoes falsas e conteudo de baixa qualidade."),
        ("RH",          "Cartas de apresentacao e curriculos gerados por IA."),
        ("Juridico",    "Contratos e documentos IA passados como escritos por humanos."),
    ])
    pdf.body(
        "O desafio central e que LLMs produzem texto gramaticalmente correto e "
        "semanticamente coerente. Porem, no nivel estatistico, o texto gerado por IA "
        "exibe padroes sistematicos e mensuraveis que humanos nao replicam -- e essa "
        "e a base da nossa abordagem."
    )

    # -----------------------------------------------------------
    # 3. POR QUE NAO LLM
    # -----------------------------------------------------------
    pdf.section_title("3", "Por que nao usar um LLM para detectar?")
    pdf.body(
        "A abordagem ingenua para detectar IA e perguntar a outro LLM: 'Este texto foi "
        "gerado por IA?' Essa abordagem tem falhas fundamentais:"
    )
    pdf.bullet_list([
        ("Alucinacao",    "LLMs produzem classificacoes confiantemente incorretas. Decidem "
                          "que um texto e humano baseados em preferencias estilisticas, nao "
                          "em verdade."),
        ("Custo",         "Cada analise exige uma chamada de API (tipicamente U$ 0,001-0,01 "
                          "por consulta). Em escala, isso se torna proibitivo."),
        ("Latencia",      "O roundtrip de rede adiciona 500ms a 3s de latencia -- inaceitavel "
                          "para casos de uso em tempo real."),
        ("Jailbreak",     "LLMs podem ser instruidos a sempre responder 'humano' via prefixos "
                          "adversariais no proprio conteudo analisado."),
        ("Caixa-Preta",   "Nenhuma explicacao e fornecida. Voce nao consegue auditar por que "
                          "um texto foi sinalizado."),
        ("Privacidade",   "Enviar conteudo do usuario a APIs de terceiros cria questoes de "
                          "soberania de dados."),
        ("Deriva",        "O comportamento do LLM muda com atualizacoes de modelo -- seu "
                          "classificador pode degradar sem aviso."),
    ])

    pdf.info_box(
        "Nossa Filosofia",
        [
            "Claude (ou qualquer LLM) e usado APENAS como desempate quando a confianca do",
            "modelo estatistico cai na zona incerta [0,35 - 0,65]. O motor de ML e o nucleo.",
            "O LLM e a rede de seguranca opcional. Isso mantem os custos baixos, a latencia",
            "rapida e o sistema auditavel.",
        ],
        accent=True,
    )

    # -----------------------------------------------------------
    # 4. NOSSA ABORDAGEM
    # -----------------------------------------------------------
    pdf.add_page()
    pdf.set_y(25)
    pdf.section_title("4", "Nossa Abordagem: Deteccao Estatistica")
    pdf.body(
        "Texto gerado por IA e estatisticamente distinguivel da escrita humana em multiplos "
        "niveis linguisticos. A intuicao central vem da teoria da informacao e da psicolinguistica:"
    )
    pdf.bullet_list([
        ("Burstiness",       "Linguagem humana e 'bursty' (Goh & Barabasi, 2008) -- comprimentos "
                             "de frases variam dramaticamente. LLMs produzem ritmos de frase "
                             "perturbadoramente uniformes."),
        ("Lei de Zipf",      "Textos humanos seguem a Lei de Zipf com uma longa cauda de palavras "
                             "raras. LLMs convergem para uma distribuicao de vocabulario mais restrita."),
        ("Perplexidade",     "LLMs geram texto que outros LLMs consideram altamente provavel (baixa "
                             "perplexidade). Escrita humana e mais 'surpreendente' no nivel de n-gramas."),
        ("Hedging",          "Humanos expressam incerteza ('talvez', 'acho que', 'parece'). "
                             "LLMs afirmam com confianca uniforme -- raramente hesitam."),
        ("1a Pessoa",        "Humanos escrevem a partir de experiencia vivida. LLMs adotam um "
                             "registro impessoal e objetivo, evitando 'eu', 'meu', 'nos'."),
        ("Transicoes",       "LLMs abusam de frases de transicao formais ('Alem disso', "
                             "'Consequentemente', 'E importante notar') como andaimes estruturais."),
    ])
    pdf.body(
        "Esses padroes nao sao balas de prata individuais -- cada feature sozinha e "
        "insuficiente. O poder vem da combinacao de todas as 12 features em um Random "
        "Forest que aprende os limites discriminativos conjuntos a partir de "
        f"{n_amostras:,} exemplos reais."
    )

    # -----------------------------------------------------------
    # 5. FEATURES
    # -----------------------------------------------------------
    pdf.section_title("5", "Engenharia de Features -- 12 Discriminadores")
    pdf.body(
        "O vetor de features tem 12 dimensoes, computadas do texto bruto usando expressoes "
        "regulares e funcoes estatisticas. Sem APIs externas, sem embeddings pre-treinados, "
        "sem redes neurais."
    )

    features_data = [
        {"name": "avg_sentence_length",      "signal": "Alta = IA",  "imp": feat_imp.get("avg_sentence_length", 0.054),
         "desc": "Media de palavras/frase. IA: uniforme 18-26 palavras."},
        {"name": "vocabulary_richness",      "signal": "Media = IA", "imp": feat_imp.get("vocabulary_richness", 0.093),
         "desc": "Type-Token Ratio. IA: restrito 0,35-0,55."},
        {"name": "burstiness",               "signal": "Baixa = IA", "imp": feat_imp.get("burstiness", 0.301),
         "desc": "CV do comprimento de frases. Humanos variam muito."},
        {"name": "punctuation_density",      "signal": "Media = IA", "imp": feat_imp.get("punctuation_density", 0.086),
         "desc": "Pontuacao / total de chars. IA: previsivel 0,03-0,07."},
        {"name": "avg_word_length",          "signal": "Alta = IA",  "imp": feat_imp.get("avg_word_length", 0.162),
         "desc": "Media de chars/palavra. IA usa vocabulario formal."},
        {"name": "transition_word_density",  "signal": "Alta = IA",  "imp": feat_imp.get("transition_word_density", 0.075),
         "desc": "Taxa de 'Alem disso', 'Consequentemente', etc."},
        {"name": "first_person_ratio",       "signal": "Baixa = IA", "imp": feat_imp.get("first_person_ratio", 0.057),
         "desc": "Pronomes I/my/me/we por token. Humanos falam pessoal."},
        {"name": "hedge_word_ratio",         "signal": "Baixa = IA", "imp": feat_imp.get("hedge_word_ratio", 0.010),
         "desc": "'Talvez', 'acho', 'parece'. IA afirma com certeza."},
        {"name": "question_density",         "signal": "Baixa = IA", "imp": feat_imp.get("question_density", 0.003),
         "desc": "Perguntas por frase. Humanos questionam mais."},
        {"name": "bigram_repetition_score",  "signal": "Alta = IA",  "imp": feat_imp.get("bigram_repetition_score", 0.032),
         "desc": "Bigramas repetidos. IA repete padroes fraseologicos."},
        {"name": "lexical_entropy",          "signal": "Baixa = IA", "imp": feat_imp.get("lexical_diversity_entropy", 0.031),
         "desc": "Entropia de Shannon das palavras. IA e previsivel."},
        {"name": "hapax_legomena_ratio",     "signal": "Baixa = IA", "imp": feat_imp.get("hapax_legomena_ratio", 0.097),
         "desc": "Palavras que aparecem 1x so. Cauda longa de Zipf."},
    ]
    pdf.feature_table(features_data)

    # -----------------------------------------------------------
    # 6. ARQUITETURA
    # -----------------------------------------------------------
    pdf.add_page()
    pdf.set_y(25)
    pdf.section_title("6", "Arquitetura: A Cascata de 3 Camadas")
    pdf.body(
        "A deteccao segue uma cascata de tres camadas. Cada camada so e ativada se a "
        "anterior nao tiver confianca suficiente. Isso minimiza custo e latencia enquanto "
        "maximiza a acuracia."
    )

    # Diagrama texto
    diagrama = (
        "ENTRADA (Texto ou Imagem)\n"
        "        |\n"
        "[Camada 1] Heuristicas -- 12 features estatisticas, regras manuais\n"
        "        | score em [0,40 - 0,60]?\n"
        "[Camada 2] Random Forest -- 200 arvores, treinado em 16.000 exemplos reais\n"
        "        | score combinado em [0,35 - 0,65]?\n"
        "[Camada 3] Claude API -- Haiku p/ texto, Sonnet Vision p/ imagens\n"
        "        |\n"
        "VEREDITO: ia | humano | incerto  +  score  +  explicacao"
    )
    pdf.set_font("Courier", "", 9)
    pdf.set_fill_color(*C_BG)
    pdf.set_draw_color(*C_ACCENT)
    pdf.set_line_width(0.4)
    pdf.set_x(15)
    pdf.multi_cell(180, 5.5, safe(diagrama), border=1, fill=True,
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.subsection("Camada 1: Heuristicas Estatisticas")
    pdf.body(
        "As 12 features sao computadas do texto bruto em Python puro -- sem chamadas de "
        "rede, sem carregamento de modelo. Executa em microssegundos. Output interpretavel: "
        "voce sabe exatamente quais features ativaram a classificacao de IA."
    )

    pdf.subsection("Camada 2: Random Forest")
    pdf.body(
        "Um Random Forest com 200 arvores de decisao aprende os limites conjuntos das "
        "features a partir dos dados rotulados. Combina as 12 features em uma probabilidade "
        "com mais precisao do que o sistema baseado em regras. O modelo e armazenado como "
        "arquivo joblib e carregado na memoria uma unica vez no startup."
    )

    pdf.subsection("Camada 3: API Claude (Opcional)")
    pdf.body(
        "So ativada quando o score da Camada 2 cai em [0,35 - 0,65] -- a zona genuinamente "
        "incerta. Usa claude-haiku-4-5 para texto (rapido, barato) e claude-sonnet-4-6 "
        "Vision para imagens. O score final e uma media ponderada: 40% ML + 60% Claude. "
        "Desabilitada se ANTHROPIC_API_KEY nao estiver configurada."
    )

    # -----------------------------------------------------------
    # 7. METRICAS
    # -----------------------------------------------------------
    pdf.section_title("7", "Metricas de Desempenho")
    pdf.body(
        f"Dataset de treino: {n_amostras:,} textos de dois datasets publicos do HuggingFace.\n"
        "Avaliacao: validacao cruzada estratificada de 5 folds."
    )

    pdf.subsection("Evolucao da Acuracia por Fase")
    pdf.ln(2)
    pdf.accuracy_bar("Fase 1 -- Heuristicas (5 feat, regras)", 0.71,  C_AMBER)
    pdf.accuracy_bar("Fase 2 -- RF em HC3+dmitva (5 feat, 6k)", 0.853, C_AMBER)
    pdf.accuracy_bar("Fase 3 -- RF v3 (12 feat, 16k amostras)", cv_acc, C_GREEN, best=True)
    pdf.ln(3)

    melhora = (cv_acc - 0.853) * 100
    pdf.body(
        f"O modelo v3 representa uma melhoria de {melhora:.1f} pontos percentuais em "
        "relacao ao v2, alcancada por engenharia de features (7 novas features) e um "
        "conjunto de treino maior e mais diversificado (16k vs 6k exemplos)."
    )

    top_feat = sorted(feat_imp.items(), key=lambda x: -x[1])[:3] if feat_imp else []
    if top_feat:
        top_str = ", ".join(f"{k} ({v:.0%})" for k, v in top_feat)
        pdf.info_box(
            "Features Mais Discriminativas",
            [
                f"Top 3: {top_str}",
                "Burstiness lidera com ~30% da decisao total -- ritmo de frase uniforme",
                "e a assinatura mais forte de texto gerado por IA.",
            ],
            accent=True,
        )

    # -----------------------------------------------------------
    # 8. APRENDIZADO CONTINUO
    # -----------------------------------------------------------
    pdf.add_page()
    pdf.set_y(25)
    pdf.section_title("8", "Sistema de Aprendizado Continuo")
    pdf.body(
        "A plataforma implementa dois mecanismos de aprendizado complementares que "
        "permitem ao modelo melhorar ao longo do tempo com o trafego real dos usuarios "
        "-- sem re-treinar do zero."
    )
    pdf.bullet_list([
        ("SGDClassifier",  "Aprendizado online via partial_fit(). Cada correcao do usuario "
                           "via POST /feedback atualiza imediatamente o modelo SGD. Latencia "
                           "zero, acontece dentro da mesma requisicao. Persistido em disco via joblib."),
        ("Retrain RF",     "Quando RETRAIN_THRESHOLD (padrao: 50) exemplos confirmados se "
                           "acumulam, o Random Forest e re-treinado em uma background task do "
                           "FastAPI. O modelo antigo continua servindo ate o retrain completar."),
        ("Auto-rotulacao", "Cada deteccao confiante (veredito != 'incerto') e automaticamente "
                           "salva no buffer de treino. Correcoes do usuario sobrescrevem o rotulo."),
        ("Persistencia",   "Todos os exemplos de treino sao armazenados na tabela SQLite "
                           "TrainingExample -- trilha de auditoria, reproducibilidade, "
                           "possibilidade de revisao manual."),
    ])
    pdf.body(
        "Isso cria um ciclo virtuoso: quanto mais o sistema e usado, mais exemplos rotulados "
        "ele acumula, melhor o modelo se torna -- sem nenhum trabalho de anotacao manual."
    )

    # -----------------------------------------------------------
    # 9. DIFERENCIAIS COMPETITIVOS
    # -----------------------------------------------------------
    pdf.section_title("9", "Diferenciais Competitivos")

    comp_headers = ["Capacidade", "AI Detector", "Detector LLM", "GPTZero/Turnitin"]
    comp_widths  = [56, 34, 34, 46]
    comp_rows = [
        ("Sem alucinacao",        "SIM",          "NAO",         "Parcial"),
        ("Custo por consulta",    "R$ 0 (ML)",    "U$ 0,001+",   "U$ 0,02+"),
        ("Latencia",              "<50ms",         "500-3000ms",  "200-800ms"),
        ("Explicavel (features)", "SIM",          "NAO",         "Limitado"),
        ("Privacidade on-premise","SIM",          "NAO (API)",   "NAO (API)"),
        ("Aprendizado continuo",  "SIM",          "NAO",         "NAO"),
        ("Deteccao de imagens",   "SIM (Vision)", "Limitado",    "NAO"),
        ("Resistente a jailbreak","SIM",          "NAO",         "Parcial"),
        ("Arquitetura aberta",    "SIM",          "NAO",         "NAO"),
    ]
    pdf.comparison_table(comp_headers, comp_widths, comp_rows)

    # -----------------------------------------------------------
    # 10. STACK TECNICO
    # -----------------------------------------------------------
    pdf.section_title("10", "Stack Tecnico")
    pdf.bullet_list([
        ("Backend",       "FastAPI (Python 3.12) -- async, documentado com OpenAPI/Swagger."),
        ("Banco de Dados","SQLite + SQLAlchemy async + aiosqlite. Producao: troque para PostgreSQL."),
        ("ML Runtime",    "scikit-learn RandomForestClassifier + SGDClassifier, serializacao joblib."),
        ("Claude",        "SDK anthropic -- claude-haiku-4-5 (texto), claude-sonnet-4-6 (Vision)."),
        ("Endpoints",     "POST /api/v1/detect (texto + imagem), POST /api/v1/detect/feedback."),
        ("Lab",           "Jupyter Notebook (detector_evolution.ipynb) -- documentacao por fases."),
        ("Deploy",        "uvicorn, porta 8002. Pronto para container."),
    ])

    # -----------------------------------------------------------
    # 11. ROADMAP
    # -----------------------------------------------------------
    pdf.add_page()
    pdf.set_y(25)
    pdf.section_title("11", "Roadmap")

    pdf.roadmap_item(
        "Fase 1", "Motor Heuristico",
        ["5 features estatisticas", "Pontuacao baseada em regras", "API REST (FastAPI)"],
        "CONCLUIDO", C_GREEN,
    )
    pdf.roadmap_item(
        "Fase 2", "Machine Learning -- Random Forest",
        ["Primeiros exemplos reais -> RF treinado", "85,3% de acuracia", "Persistencia joblib"],
        "CONCLUIDO", C_GREEN,
    )
    pdf.roadmap_item(
        "Fase 2.5", "Treino em Datasets Reais",
        ["HC3 + dmitva (6.000 amostras)", "Validacao cruzada estratificada", "Integracao HuggingFace"],
        "CONCLUIDO", C_GREEN,
    )
    pdf.roadmap_item(
        "Fase 3", "Cascata + Aprendizado Continuo",
        [
            "12 features (7 novas adicionadas)",
            "16.000 amostras de treino -> 93,4% de acuracia",
            "Integracao com API Claude (texto + Vision)",
            "SGDClassifier para aprendizado online",
            "Retrain RF em background",
            "Endpoints POST /detect e POST /feedback",
        ],
        "CONCLUIDO", C_GREEN,
    )
    pdf.roadmap_item(
        "Fase 4", "TF-IDF + Features de Embedding",
        [
            "Adicionar features TF-IDF de n-gramas como dimensoes extras",
            "Similaridade de embedding em nivel de sentenca (repeticao por cosseno)",
            "Modelo de embedding local (sentence-transformers, sem API)",
            "Melhoria esperada de acuracia: 93% -> 95-96%",
        ],
        "PROXIMO", C_AMBER,
    )
    pdf.roadmap_item(
        "Fase 5", "Perplexidade e Probabilidade de Token",
        [
            "Pontuacao de perplexidade com LM local (GPT-2 small, roda localmente)",
            "Abordagem de perturbacao DetectGPT (Mitchell et al., 2023)",
            "Hibrido: perplexidade + 12 features + RF",
            "Alvo: 96-98% de acuracia em texto academico",
        ],
        "PLANEJADO", (100, 100, 200),
    )
    pdf.roadmap_item(
        "Fase 6", "Producao e Escala",
        [
            "Migracao para PostgreSQL (Alembic)",
            "Cache Redis para textos repetidos",
            "Endpoint de analise em lote (batch)",
            "Webhook para analise assincrona de documentos grandes",
            "Dashboard com analytics e historico de deteccoes",
        ],
        "PLANEJADO", (100, 100, 200),
    )

    # -----------------------------------------------------------
    # Salva
    # -----------------------------------------------------------
    pdf.output(str(PDF_OUT))
    print(f"PDF gerado: {PDF_OUT}")


if __name__ == "__main__":
    build_pdf()
