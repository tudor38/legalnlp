#!/usr/bin/env python3
"""
Generate fictitious Word documents (.docx) for testing the wordnlp app.

Creates:
  test_docs/services_agreement.docx  — Software Development Services Agreement
  test_docs/nda.docx                 — Mutual Non-Disclosure Agreement

Features across both documents:
  • Rich contract text with specific dollar amounts, dates, percentages, SLAs
  • Tracked changes (insertions/deletions) by multiple authors
  • Two sets of moved paragraphs (moveFrom / moveTo) per document
  • 25 comments / 8 threads in services_agreement.docx
  • 15 comments / 5 threads in nda.docx
  • Reply chains up to 3 levels
  • Date range ~8 weeks per document

Usage:
    python scripts/make_test_docs.py
"""

import os
import zipfile

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_docs")

# ---------------------------------------------------------------------------
# Low-level XML helpers
# ---------------------------------------------------------------------------


def _xe(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _t(text: str) -> str:
    if text != text.strip():
        return f'<w:t xml:space="preserve">{_xe(text)}</w:t>'
    return f"<w:t>{_xe(text)}</w:t>"


def _dt(text: str) -> str:
    if text != text.strip():
        return f'<w:delText xml:space="preserve">{_xe(text)}</w:delText>'
    return f"<w:delText>{_xe(text)}</w:delText>"


# ---------------------------------------------------------------------------
# Run-level XML builders
# ---------------------------------------------------------------------------


def run(text: str, bold: bool = False, italic: bool = False) -> str:
    rpr = ""
    if bold or italic:
        rpr = (
            "<w:rPr>"
            + ("<w:b/>" if bold else "")
            + ("<w:i/>" if italic else "")
            + "</w:rPr>"
        )
    return f"<w:r>{rpr}{_t(text)}</w:r>"


def ins(text: str, author: str, date: str, rid: int) -> str:
    return (
        f'<w:ins w:id="{rid}" w:author="{_xe(author)}" w:date="{date}">'
        f"<w:r>{_t(text)}</w:r>"
        f"</w:ins>"
    )


def del_(text: str, author: str, date: str, rid: int) -> str:
    return (
        f'<w:del w:id="{rid}" w:author="{_xe(author)}" w:date="{date}">'
        f"<w:r>{_dt(text)}</w:r>"
        f"</w:del>"
    )


def ins_para(author: str, date: str, rid: int, content: str) -> str:
    return (
        f'<w:ins w:id="{rid}" w:author="{_xe(author)}" w:date="{date}">'
        f"{content}"
        f"</w:ins>"
    )


def move_from_block(paras: str, author: str, date: str, rid: int) -> str:
    return (
        f'<w:moveFrom w:id="{rid}" w:author="{_xe(author)}" w:date="{date}">'
        f"{paras}"
        f"</w:moveFrom>"
    )


def move_to_block(paras: str, author: str, date: str, rid: int) -> str:
    return (
        f'<w:moveTo w:id="{rid}" w:author="{_xe(author)}" w:date="{date}">'
        f"{paras}"
        f"</w:moveTo>"
    )


def cstart(cid: int) -> str:
    return f'<w:commentRangeStart w:id="{cid}"/>'


def cend(cid: int) -> str:
    return f'<w:commentRangeEnd w:id="{cid}"/>'


def cref(cid: int) -> str:
    return (
        f"<w:r>"
        f'<w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
        f'<w:commentReference w:id="{cid}"/>'
        f"</w:r>"
    )


def crefs(*cids: int) -> str:
    return "".join(cref(c) for c in cids)


# ---------------------------------------------------------------------------
# Paragraph-level XML builders
# ---------------------------------------------------------------------------


def para(content: str, pid: str = "") -> str:
    pid_attr = f' w14:paraId="{pid}"' if pid else ""
    return f"<w:p{pid_attr}>{content}</w:p>"


def title_para(text: str, pid: str = "") -> str:
    pid_attr = f' w14:paraId="{pid}"' if pid else ""
    return (
        f"<w:p{pid_attr}>"
        f'<w:pPr><w:jc w:val="center"/>'
        f'<w:rPr><w:b/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:pPr>'
        f'<w:r><w:rPr><w:b/><w:sz w:val="32"/></w:rPr>{_t(text)}</w:r>'
        f"</w:p>"
    )


def heading(text: str, pid: str = "") -> str:
    pid_attr = f' w14:paraId="{pid}"' if pid else ""
    return (
        f"<w:p{pid_attr}>"
        f'<w:pPr><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:pPr>'
        f'<w:r><w:rPr><w:b/><w:sz w:val="26"/></w:rPr>{_t(text)}</w:r>'
        f"</w:p>"
    )


def blank() -> str:
    return "<w:p><w:r><w:t></w:t></w:r></w:p>"


# ---------------------------------------------------------------------------
# Comment XML builders
# ---------------------------------------------------------------------------


def comment_xml(
    cid: int, author: str, date: str, initials: str, text: str, para_id: str
) -> str:
    return (
        f'<w:comment w:id="{cid}" w:author="{_xe(author)}" '
        f'w:date="{date}" w:initials="{_xe(initials)}">'
        f'<w:p w14:paraId="{para_id}" w14:textId="FFFFFFFF">'
        f'<w:pPr><w:pStyle w:val="CommentText"/></w:pPr>'
        f'<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
        f"<w:annotationRef/></w:r>"
        f"<w:r>{_t(text)}</w:r>"
        f"</w:p>"
        f"</w:comment>"
    )


# ---------------------------------------------------------------------------
# Document parts
# ---------------------------------------------------------------------------


def content_types() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '<Override PartName="/word/comments.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>'
        '<Override PartName="/word/commentsExtended.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.commentsExtended+xml"/>'
        '<Override PartName="/word/commentsIds.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.commentsIds+xml"/>'
        "</Types>"
    )


ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/>'
    "</Relationships>"
)

DOC_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
    'Target="styles.xml"/>'
    '<Relationship Id="rId2" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" '
    'Target="comments.xml"/>'
    '<Relationship Id="rId3" '
    'Type="http://schemas.microsoft.com/office/2011/relationships/commentsExtended" '
    'Target="commentsExtended.xml"/>'
    '<Relationship Id="rId4" '
    'Type="http://schemas.microsoft.com/office/2016/09/relationships/commentsIds" '
    'Target="commentsIds.xml"/>'
    "</Relationships>"
)

STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
    '<w:name w:val="Normal"/>'
    '<w:rPr><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
    "</w:style>"
    '<w:style w:type="character" w:styleId="CommentReference">'
    '<w:name w:val="Comment Reference"/>'
    '<w:rPr><w:sz w:val="16"/></w:rPr>'
    "</w:style>"
    '<w:style w:type="paragraph" w:styleId="CommentText">'
    '<w:name w:val="Comment Text"/>'
    "<w:pPr/>"
    '<w:rPr><w:sz w:val="20"/></w:rPr>'
    "</w:style>"
    "</w:styles>"
)

DOC_NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
    'xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml"'
)


def wrap_document(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<w:document {DOC_NS}>"
        f"<w:body>{body}"
        "<w:sectPr/>"
        "</w:body>"
        "</w:document>"
    )


def wrap_comments(inner: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<w:comments {DOC_NS}>"
        f"{inner}"
        "</w:comments>"
    )


def wrap_comments_extended(inner: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        "<w15:commentsEx "
        'xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml">'
        f"{inner}"
        "</w15:commentsEx>"
    )


def comment_ex(para_id: str, parent_para_id: str = "", done: str = "0") -> str:
    parent_attr = f' w15:paraIdParent="{parent_para_id}"' if parent_para_id else ""
    return f'<w15:commentEx w15:paraId="{para_id}" w15:done="{done}"{parent_attr}/>'


def wrap_comments_ids(inner: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        "<w16cid:commentsIds "
        'xmlns:w16cid="http://schemas.microsoft.com/office/word/2016/wordml/cid">'
        f"{inner}"
        "</w16cid:commentsIds>"
    )


def comment_id_entry(para_id: str, durable_id: int) -> str:
    return (
        f'<w16cid:commentId w16cid:paraId="{para_id}" w16cid:durableId="{durable_id}"/>'
    )


# ---------------------------------------------------------------------------
# Helper to write a .docx zip
# ---------------------------------------------------------------------------


def write_docx(
    path: str,
    document_xml: str,
    comments_xml: str,
    comments_extended_xml: str,
    comments_ids_xml: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types())
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("word/_rels/document.xml.rels", DOC_RELS)
        z.writestr("word/styles.xml", STYLES)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/comments.xml", comments_xml)
        z.writestr("word/commentsExtended.xml", comments_extended_xml)
        z.writestr("word/commentsIds.xml", comments_ids_xml)
    print(f"  wrote {path}")


# ===========================================================================
# DOCUMENT 1: Software Development Services Agreement
# ===========================================================================
#
# Parties:    Acme Corporation ("Client")  vs  TechVenture Ltd. ("Service Provider")
# Authors:    Sarah Chen (Acme counsel)  |  Marcus Webb (TechVenture counsel)
#             Jennifer Park (Acme VP Engineering)  |  David Okafor (TechVenture CEO)
# Contract value: USD 150,000 fixed fee + USD 8,500/month support retainer
# Date range: 2024-01-15 → 2024-02-28
#
# Track changes (21 redlines):
#   1/2:   Scope narrowed: "develop, test, deploy, and maintain" → "develop and deliver"  (Sarah, Jan 20)
#   3:     MOVE 1 — Deliverables section moved from §1 to §11  (Marcus, Jan 25)
#   4:     MOVE 2 — Data Protection para moved from §2 to §12  (Sarah, Feb 20)
#   5/6/7: Payment: "forty-five (45) business" → "thirty (30) calendar" days  (Sarah, Jan 22)
#   8/9:   IP: "jointly in both parties" → "solely in Client"  (Sarah, Jan 20)
#   10:    IP carve-out paragraph inserted  (Marcus, Jan 24)
#   11/12: Confidentiality: "five (5)" → "two (2)" years  (Marcus, Feb 01)
#   13/14: Termination notice: "ninety (90)" → "thirty (30)" days  (Sarah, Feb 03)
#   15/16: Warranty period: "twelve (12) months" → "six (6) months"  (Marcus, Feb 08)
#   17/18: Liability cap: "$1,000,000" → "$500,000"  (Marcus, Feb 15)
#   19:    Indemnification carve-out inserted  (Sarah, Feb 20)
#   20/21: Phase 3 fee: "$50,000" → "$45,000"  (Sarah, Jan 28)
#
# Comments (25 total / 8 threads):
#   C1   Marcus / Jan 15   — effective date clarification (standalone, constructive)
#   C2   Marcus / Jan 15   — "necessary resources" scope concern (standalone, diplomatic)
#   C3   Sarah  / Jan 22   — 30-day payment non-negotiable (root thread A, blunt)
#   C4   Marcus / Jan 23   — calendar vs business days question (reply A)
#   C5   Sarah  / Jan 23   — confirms calendar days (reply A)
#   C6   Sarah  / Jan 20   — IP must be "solely" Client (root thread B, demanding)
#   C7   Marcus / Jan 24   — IP carve-out pushback (reply B, firm)
#   C8   Jennifer / Feb 01 — confidentiality duration question (root thread C)
#   C9   Sarah  / Feb 02   — 2-year compromise rationale (reply C, diplomatic)
#   C10  David  / Feb 03   — termination 30 days unworkable (root thread D, blunt)
#   C11  Sarah  / Feb 05   — 60-day compromise offer (reply D, diplomatic)
#   C12  David  / Feb 06   — accepts 60 days (reply D, brief)
#   C13  Marcus / Feb 10   — governing law NY vs DE (root thread E, pushback)
#   C14  Sarah  / Feb 11   — NY non-negotiable (reply E, blunt)
#   C15  Marcus / Feb 12   — accepts NY, wants arbitration (reply E, conceding)
#   C16  Jennifer / Jan 16 — milestone schedule question (standalone, constructive)
#   C17  David  / Jan 18   — objects to Phase 3 fee cut (root thread F, blunt)
#   C18  Sarah  / Jan 19   — defends Phase 3 reduction (reply F)
#   C19  Marcus / Jan 20   — accepts with Phase 2 adjustment (reply F, constructive)
#   C20  Sarah  / Feb 08   — warranty 6 months too short (root thread G, blunt)
#   C21  Marcus / Feb 09   — rationale for 6 months (reply G, diplomatic)
#   C22  Sarah  / Feb 15   — $500K cap dangerously low (root thread H, blunt)
#   C23  Marcus / Feb 16   — defends $500K cap (reply H)
#   C24  David  / Feb 17   — supports Marcus on cap (reply H)
#   C25  Jennifer / Feb 22 — final sign-off, all points closed (standalone, constructive)
# ===========================================================================


def make_services_agreement() -> None:

    body = ""

    # ── Title ────────────────────────────────────────────────────────────────
    body += title_para("SOFTWARE DEVELOPMENT SERVICES AGREEMENT")
    body += blank()

    # ── Preamble  [C1 on effective date] ─────────────────────────────────────
    body += para(
        run(
            "This Software Development Services Agreement (the \u201cAgreement\u201d) "
            "is entered into as of "
        )
        + cstart(1)
        + run("January 15, 2024")
        + cend(1)
        + cref(1)
        + run(
            " (the \u201cEffective Date\u201d), between Acme Corporation, a New York "
            "corporation with its principal place of business at 350 Fifth Avenue, "
            "New York, NY 10118 (\u201cClient\u201d), and TechVenture Ltd., a "
            "Delaware corporation with its principal place of business at 1 Market "
            "Street, Suite 900, San Francisco, CA 94105 (\u201cService Provider\u201d) "
            "(together, the \u201cParties\u201d). The total fixed fee under this "
            "Agreement is USD 150,000, payable in milestones as set out in "
            "Section\u00a02. This Agreement supersedes all prior negotiations and "
            "the Letter of Intent dated December 1, 2023."
        ),
        pid="10000001",
    )
    body += blank()

    # ── §1 SERVICES  [TC 1/2: scope; C2 on "necessary resources"] ───────────
    body += heading("1.  SERVICES", pid="10000002")
    body += para(
        run("Service Provider agrees to ")
        + del_(
            "develop, test, deploy, and maintain",
            "Sarah Chen",
            "2024-01-20T09:00:00Z",
            1,
        )
        + ins("develop and deliver", "Sarah Chen", "2024-01-20T09:00:00Z", 2)
        + run(
            " a custom inventory management software system (the \u201cSoftware\u201d) "
            "for Client\u2019s warehouse operations across its three (3) distribution "
            "centres located in Newark (NJ), Columbus (OH), and Atlanta (GA). "
            "The Software shall integrate with Client\u2019s existing SAP ERP platform "
            "(version 4.7) via REST API and shall support a minimum of 500 concurrent "
            "users with a page-load response time not exceeding 2\u00a0seconds under "
            "normal operating conditions. "
        )
        + cstart(2)
        + run(
            "Service Provider shall provide all necessary personnel, equipment, "
            "and resources required to complete the Services."
        )
        + cend(2)
        + cref(2),
        pid="10000003",
    )
    body += blank()

    # ── MOVE 1: Deliverables — MOVED FROM HERE ────────────────────────────────
    body += move_from_block(
        heading("2.  DELIVERABLES", pid="10000004")
        + para(
            run(
                "Service Provider shall deliver the Software in three (3) phases "
                "as set out in the project schedule attached as Exhibit\u00a0B. "
                "Phase\u00a01 (Requirements & Architecture) shall be completed by "
                "March 31, 2024; Phase\u00a02 (Development & Integration) by "
                "June 30, 2024; Phase\u00a03 (UAT & Go-Live) by September 30, 2024. "
                "Each phase shall conclude with a written acceptance sign-off from "
                "Client within ten (10) business days of delivery. Failure to provide "
                "written rejection within that period shall constitute deemed acceptance."
            ),
            pid="10000005",
        ),
        "Marcus Webb",
        "2024-01-25T14:00:00Z",
        3,
    )
    body += blank()

    # ── §2 FEE SCHEDULE  [TC 20/21 on Phase 3; C16 on total; C17/18/19 on Phase 3] ──
    body += heading("2.  FEE SCHEDULE", pid="10000006")
    body += para(
        cstart(16)
        + run(
            "The fixed fee of USD\u00a0150,000 is payable in three (3) milestone "
            "instalments as follows:"
        )
        + cend(16)
        + cref(16),
        pid="10000007",
    )
    body += para(
        run(
            "Phase\u00a01 — Requirements & Architecture: USD\u00a045,000, due within "
            "fifteen (15) calendar days of Client\u2019s written acceptance of the "
            "Phase\u00a01 deliverables."
        ),
        pid="10000008",
    )
    body += para(
        run(
            "Phase\u00a02 — Development & Integration: USD\u00a060,000, due within "
            "fifteen (15) calendar days of Client\u2019s written acceptance of the "
            "Phase\u00a02 deliverables."
        ),
        pid="10000009",
    )
    body += para(
        run("Phase\u00a03 — UAT & Go-Live: ")
        + del_("USD\u00a050,000", "Sarah Chen", "2024-01-28T10:00:00Z", 20)
        + cstart(17)
        + ins("USD\u00a045,000", "Sarah Chen", "2024-01-28T10:00:00Z", 21)
        + cend(17)
        + crefs(17, 18, 19)
        + run(
            ", due within fifteen (15) calendar days of Client\u2019s written "
            "acceptance of the final deliverables and successful completion of "
            "user acceptance testing (\u201cUAT\u201d)."
        ),
        pid="1000000A",
    )
    body += para(
        run(
            "In addition to the fixed fee, Client shall pay Service Provider a "
            "monthly support retainer of USD\u00a08,500 per month for a period of "
            "twelve (12) months following the Go-Live date (total support fee: "
            "USD\u00a0102,000). The retainer covers up to forty (40) hours of "
            "maintenance, bug-fixes, and minor enhancements per month. Hours in "
            "excess of forty (40) shall be billed at USD\u00a0175 per hour."
        ),
        pid="1000000B",
    )
    body += blank()

    # ── MOVE 2: Data Protection para — MOVED FROM HERE ───────────────────────
    body += move_from_block(
        para(
            run(
                "Service Provider shall implement and maintain appropriate technical "
                "and organisational measures to protect Client\u2019s data processed "
                "under this Agreement, including at minimum: AES-256 encryption at "
                "rest and in transit; role-based access controls; annual penetration "
                "testing by a qualified third party; and incident response procedures "
                "capable of notifying Client within 72\u00a0hours of discovery of any "
                "personal data breach. Service Provider shall not transfer Client data "
                "outside the United States without Client\u2019s prior written consent."
            ),
            pid="1000000C",
        ),
        "Sarah Chen",
        "2024-02-20T11:00:00Z",
        4,
    )
    body += blank()

    # ── §3 PAYMENT TERMS  [TC 5/6/7; C3/C4/C5] ──────────────────────────────
    body += heading("3.  PAYMENT TERMS", pid="1000000D")
    body += para(
        run("Client shall pay Service Provider\u2019s invoices within ")
        + del_("forty-five (45) business", "Sarah Chen", "2024-01-22T14:00:00Z", 5)
        + ins(" ", "Sarah Chen", "2024-01-22T14:00:00Z", 6)
        + cstart(3)
        + ins("thirty (30) calendar", "Sarah Chen", "2024-01-22T14:00:00Z", 7)
        + cend(3)
        + crefs(3, 4, 5)
        + run(
            " days of receipt of a valid invoice. Late payments shall accrue "
            "interest at the rate of 1.5\u00a0% per month (18\u00a0% per annum) "
            "on the outstanding balance from the due date until the date of actual "
            "payment. Service Provider may suspend performance on five (5) business "
            "days\u2019 written notice if any invoice exceeding USD\u00a05,000 "
            "remains unpaid after its due date. All fees are exclusive of applicable "
            "sales, use, or value-added taxes, which shall be borne by Client. "
            "Invoices shall be submitted electronically to accounts.payable@acmecorp.com."
        ),
        pid="1000000E",
    )
    body += blank()

    # ── §4 INTELLECTUAL PROPERTY  [TC 8/9/10; C6/C7] ────────────────────────
    body += heading("4.  INTELLECTUAL PROPERTY", pid="1000000F")
    body += para(
        run(
            "All intellectual property rights in any work product, deliverables, "
            "source code, documentation, or software created by Service Provider "
            "specifically for Client under this Agreement shall vest "
        )
        + del_("jointly in both parties", "Sarah Chen", "2024-01-20T09:00:00Z", 8)
        + cstart(6)
        + ins("solely in Client", "Sarah Chen", "2024-01-20T09:00:00Z", 9)
        + cend(6)
        + crefs(6, 7)
        + run(
            " upon receipt of full payment of all outstanding invoices. Service "
            "Provider hereby assigns, and agrees to cause its personnel to assign, "
            "all such intellectual property rights to Client and shall execute any "
            "documents necessary to perfect that assignment, including any patent "
            "assignment deeds and copyright transfer instruments."
        ),
        pid="10000010",
    )
    # IP carve-out paragraph inserted by Marcus [TC 10]
    body += para(
        ins(
            "Notwithstanding the foregoing, Service Provider retains all rights in "
            "and to its pre-existing intellectual property, proprietary tools, "
            "frameworks, libraries, and general methodologies developed independently "
            "of this Agreement (\u201cBackground IP\u201d). A non-exhaustive list of "
            "Background IP is set out in Exhibit\u00a0C. Client is granted a perpetual, "
            "non-exclusive, royalty-free licence to use Background IP solely to the "
            "extent embedded in or necessary to operate the deliverables.",
            "Marcus Webb",
            "2024-01-24T11:30:00Z",
            10,
        ),
        pid="10000011",
    )
    body += blank()

    # ── §5 WARRANTIES  [TC 15/16; C20/C21] ───────────────────────────────────
    body += heading("5.  WARRANTIES", pid="10000012")
    body += para(
        run(
            "Service Provider warrants that: (a)\u00a0the Software will perform "
            "materially in accordance with the functional specifications in Exhibit\u00a0A "
            "for a period of "
        )
        + del_("twelve (12) months", "Marcus Webb", "2024-02-08T10:00:00Z", 15)
        + cstart(20)
        + ins("six (6) months", "Marcus Webb", "2024-02-08T10:00:00Z", 16)
        + cend(20)
        + crefs(20, 21)
        + run(
            " following Go-Live (\u201cWarranty Period\u201d); "
            "(b)\u00a0the Software will not contain any malicious code, viruses, "
            "or undisclosed disabling mechanisms; (c)\u00a0Service Provider has "
            "full right and authority to enter into this Agreement and to grant "
            "the rights granted herein; and (d)\u00a0the Software, when delivered, "
            "will not infringe any third-party intellectual property rights. During "
            "the Warranty Period, Service Provider shall remedy any material defect "
            "within ten (10) business days of written notice at no additional charge. "
            "EXCEPT AS SET OUT IN THIS SECTION, THE SOFTWARE IS PROVIDED \u201cAS "
            "IS\u201d AND ALL OTHER WARRANTIES ARE DISCLAIMED."
        ),
        pid="10000013",
    )
    body += blank()

    # ── §6 LIMITATION OF LIABILITY  [TC 17/18; C22/C23/C24] ─────────────────
    body += heading("6.  LIMITATION OF LIABILITY", pid="10000014")
    body += para(
        run(
            "TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, NEITHER PARTY "
            "SHALL BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, SPECIAL, "
            "PUNITIVE, OR CONSEQUENTIAL DAMAGES, INCLUDING LOSS OF PROFITS, LOSS "
            "OF DATA, OR LOSS OF BUSINESS, EVEN IF ADVISED OF THE POSSIBILITY OF "
            "SUCH DAMAGES. IN NO EVENT SHALL SERVICE PROVIDER\u2019S AGGREGATE "
            "LIABILITY UNDER OR RELATING TO THIS AGREEMENT EXCEED THE GREATER OF "
            "(A)\u00a0"
        )
        + del_("USD\u00a01,000,000", "Marcus Webb", "2024-02-15T09:00:00Z", 17)
        + cstart(22)
        + ins("USD\u00a0500,000", "Marcus Webb", "2024-02-15T09:00:00Z", 18)
        + cend(22)
        + crefs(22, 23, 24)
        + run(
            " OR (B)\u00a0THE TOTAL FEES ACTUALLY PAID BY CLIENT TO SERVICE "
            "PROVIDER IN THE TWELVE (12) MONTHS IMMEDIATELY PRECEDING THE EVENT "
            "GIVING RISE TO THE CLAIM. THE PARTIES ACKNOWLEDGE THAT THESE "
            "LIMITATIONS REFLECT A REASONABLE ALLOCATION OF RISK AND FORM AN "
            "ESSENTIAL BASIS OF THE BARGAIN BETWEEN THEM."
        ),
        pid="10000015",
    )
    body += blank()

    # ── §7 INDEMNIFICATION  [TC 19] ───────────────────────────────────────────
    body += heading("7.  INDEMNIFICATION", pid="10000016")
    body += para(
        run(
            "Each party (\u201cIndemnifying Party\u201d) shall indemnify, defend, "
            "and hold harmless the other party and its officers, directors, employees, "
            "and agents (\u201cIndemnified Parties\u201d) from and against any "
            "third-party claims, damages, losses, and expenses (including reasonable "
            "legal fees) arising out of or relating to: (a)\u00a0the Indemnifying "
            "Party\u2019s breach of any representation, warranty, or obligation under "
            "this Agreement; or (b)\u00a0the Indemnifying Party\u2019s gross negligence "
            "or wilful misconduct. "
        )
        + ins(
            "Notwithstanding the foregoing, Service Provider\u2019s indemnification "
            "obligation shall not apply to any claim arising from Client\u2019s "
            "modification of the Software without Service Provider\u2019s prior "
            "written approval, or from Client\u2019s use of the Software in a manner "
            "not contemplated by the technical specifications in Exhibit\u00a0A.",
            "Sarah Chen",
            "2024-02-20T14:00:00Z",
            19,
        ),
        pid="10000017",
    )
    body += blank()

    # ── §8 CONFIDENTIALITY  [TC 11/12; C8/C9] ────────────────────────────────
    body += heading("8.  CONFIDENTIALITY", pid="10000018")
    body += para(
        run(
            "Each party (\u201cReceiving Party\u201d) agrees to hold in strict "
            "confidence all non-public, proprietary information of the other party "
            "(\u201cDisclosing Party\u201d) disclosed in connection with this "
            "Agreement (\u201cConfidential Information\u201d) and to use such "
            "information solely for performing its obligations or exercising its "
            "rights hereunder. This obligation shall survive termination for a "
            "period of "
        )
        + del_("five (5)", "Marcus Webb", "2024-02-01T10:00:00Z", 11)
        + cstart(8)
        + ins("two (2)", "Marcus Webb", "2024-02-01T10:00:00Z", 12)
        + cend(8)
        + crefs(8, 9)
        + run(
            " years. Confidential Information does not include information that: "
            "(i)\u00a0is or becomes publicly available through no fault of the "
            "Receiving Party; (ii)\u00a0was rightfully known to the Receiving Party "
            "before disclosure; (iii)\u00a0is independently developed by the "
            "Receiving Party without reference to the Confidential Information; or "
            "(iv)\u00a0is required to be disclosed by applicable law or court order, "
            "provided the Receiving Party gives prompt written notice to the "
            "Disclosing Party and cooperates in seeking a protective order."
        ),
        pid="10000019",
    )
    body += blank()

    # ── §9 TERM AND TERMINATION  [TC 13/14; C10/C11/C12] ────────────────────
    body += heading("9.  TERM AND TERMINATION", pid="1000001A")
    body += para(
        run(
            "This Agreement commences on the Effective Date and continues until "
            "the later of: (a)\u00a0completion of all deliverables and acceptance "
            "by Client; or (b)\u00a0expiry of the support retainer period, unless "
            "terminated earlier pursuant to this Section. Either party may terminate "
            "this Agreement for convenience upon "
        )
        + del_("ninety (90)", "Sarah Chen", "2024-02-03T09:00:00Z", 13)
        + cstart(10)
        + ins("thirty (30)", "Sarah Chen", "2024-02-03T09:00:00Z", 14)
        + cend(10)
        + crefs(10, 11, 12)
        + run(
            " days\u2019 prior written notice to the other party. Either party may "
            "terminate immediately for cause upon written notice if the other party "
            "commits a material breach and fails to cure such breach within fifteen "
            "(15) business days of receiving written notice specifying the breach in "
            "reasonable detail. Upon termination for any reason, each party shall "
            "promptly return or destroy the other party\u2019s Confidential "
            "Information and, upon request, certify such destruction in writing. "
            "Sections 4 (Intellectual Property), 6 (Limitation of Liability), "
            "7 (Indemnification), 8 (Confidentiality), and 10 (Governing Law) "
            "shall survive termination."
        ),
        pid="1000001B",
    )
    body += blank()

    # ── §10 GOVERNING LAW  [C13/C14/C15] ─────────────────────────────────────
    body += heading("10.  GOVERNING LAW", pid="1000001C")
    body += para(
        run(
            "This Agreement shall be governed by and construed in accordance with "
            "the laws of the "
        )
        + cstart(13)
        + run("State of New York")
        + cend(13)
        + crefs(13, 14, 15)
        + run(
            ", without regard to its conflict-of-laws principles. Each party "
            "irrevocably submits to the exclusive jurisdiction of the state and "
            "federal courts sitting in New York County, New York for the resolution "
            "of any dispute arising under or relating to this Agreement. Process "
            "may be served by any means authorised under the laws of the "
            "State of New York."
        ),
        pid="1000001D",
    )
    body += blank()

    # ── §11 DELIVERABLES — MOVED TO HERE (MOVE 1) ────────────────────────────
    body += heading("11.  DELIVERABLES", pid="1000001E")
    body += move_to_block(
        para(
            run(
                "Service Provider shall deliver the Software in three (3) phases "
                "as set out in the project schedule attached as Exhibit\u00a0B. "
                "Phase\u00a01 (Requirements & Architecture) shall be completed by "
                "March 31, 2024; Phase\u00a02 (Development & Integration) by "
                "June 30, 2024; Phase\u00a03 (UAT & Go-Live) by September 30, 2024. "
                "Each phase shall conclude with a written acceptance sign-off from "
                "Client within ten (10) business days of delivery. Failure to provide "
                "written rejection within that period shall constitute deemed acceptance."
            ),
            pid="1000001F",
        ),
        "Marcus Webb",
        "2024-01-25T14:00:00Z",
        3,
    )
    body += blank()

    # ── §12 DATA PROTECTION — MOVED TO HERE (MOVE 2) ─────────────────────────
    body += heading("12.  DATA PROTECTION", pid="10000020")
    body += move_to_block(
        para(
            run(
                "Service Provider shall implement and maintain appropriate technical "
                "and organisational measures to protect Client\u2019s data processed "
                "under this Agreement, including at minimum: AES-256 encryption at "
                "rest and in transit; role-based access controls; annual penetration "
                "testing by a qualified third party; and incident response procedures "
                "capable of notifying Client within 72\u00a0hours of discovery of any "
                "personal data breach. Service Provider shall not transfer Client data "
                "outside the United States without Client\u2019s prior written consent."
            ),
            pid="10000021",
        ),
        "Sarah Chen",
        "2024-02-20T11:00:00Z",
        4,
    )
    body += blank()

    # ── §13 GENERAL PROVISIONS  [C25 standalone] ─────────────────────────────
    body += heading("13.  GENERAL PROVISIONS", pid="10000022")
    body += para(
        cstart(25)
        + run(
            "This Agreement, together with all Exhibits, constitutes the entire "
            "agreement between the Parties with respect to its subject matter and "
            "supersedes all prior agreements, representations, and understandings "
            "between the Parties, including the Letter of Intent dated December\u00a01, "
            "2023. Any amendment must be in writing and signed by authorised "
            "representatives of both Parties. If any provision is held invalid or "
            "unenforceable, the remaining provisions shall continue in full force "
            "and effect. Neither party may assign this Agreement without the other "
            "party\u2019s prior written consent (not to be unreasonably withheld or "
            "delayed), except that either party may assign to an affiliate or to a "
            "successor in connection with a merger or acquisition of all or "
            "substantially all of its assets. Notices under this Agreement shall be "
            "in writing and delivered by email with read-receipt confirmation or "
            "by overnight courier to the addresses set out in the signature block."
        )
        + cend(25)
        + cref(25),
        pid="10000023",
    )

    document_xml = wrap_document(body)

    # ── Comments ──────────────────────────────────────────────────────────────

    c = ""

    # C1 standalone — effective date (Marcus, Jan 15, constructive)
    c += comment_xml(
        1,
        "Marcus Webb",
        "2024-01-15T10:30:00Z",
        "MW",
        "Quick process note \u2014 to confirm: the Effective Date is tied to the "
        "date of final countersignature by the last party to sign, not the calendar "
        "date written above? Acme\u2019s finance team flagged a discrepancy on this "
        "in our last engagement and we want to be explicit. Suggest adding: "
        "\u201c\u2026the date last signed below.\u201d",
        "C0000001",
    )

    # C2 standalone — necessary resources (Marcus, Jan 15, diplomatic)
    c += comment_xml(
        2,
        "Marcus Webb",
        "2024-01-15T11:00:00Z",
        "MW",
        "I\u2019d recommend we either define \u201cnecessary\u201d by reference to "
        "the staffing schedule in Exhibit\u00a0B, or add an explicit cost cap. "
        "As currently drafted this clause could be read as requiring Service Provider "
        "to bear unlimited procurement costs, which is not the commercial intent. "
        "Happy to suggest language if helpful.",
        "C0000002",
    )

    # Thread A: payment terms (C3 root, C4 reply, C5 reply)
    c += comment_xml(
        3,
        "Sarah Chen",
        "2024-01-22T14:00:00Z",
        "SC",
        "Thirty days was expressly agreed in the LOI (paragraph 4) and is "
        "non-negotiable from Acme\u2019s side. I have updated the clause accordingly. "
        "Do not revert this.",
        "C0000003",
    )
    c += comment_xml(
        4,
        "Marcus Webb",
        "2024-01-23T09:15:00Z",
        "MW",
        "No objection to 30 days \u2014 understood and accepted. My only question: "
        "are these calendar days or business days? The LOI used the term "
        "\u201cdays\u201d without qualification. We are fine with calendar days "
        "provided that is what the text says.",
        "C0000004",
    )
    c += comment_xml(
        5,
        "Sarah Chen",
        "2024-01-23T10:00:00Z",
        "SC",
        "Calendar days, confirmed. I have added \u201ccalendar\u201d to the clause. "
        "This point is closed.",
        "C0000005",
    )

    # Thread B: IP ownership (C6 root, C7 reply)
    c += comment_xml(
        6,
        "Sarah Chen",
        "2024-01-20T09:00:00Z",
        "SC",
        "This MUST read \u201csolely in Client.\u201d Co-ownership is commercially "
        "unworkable \u2014 it prevents Client from licensing, enforcing, or "
        "sublicensing the Software without TechVenture\u2019s consent. This has "
        "been litigated in multiple jurisdictions and we will not accept \u201cjointly.\u201d "
        "Do NOT soften this language.",
        "C0000006",
    )
    c += comment_xml(
        7,
        "Marcus Webb",
        "2024-01-24T11:30:00Z",
        "MW",
        "We understand Acme\u2019s position but a blanket assignment of \u201call IP\u201d "
        "is not acceptable. TechVenture has 12\u00a0years of proprietary algorithms, "
        "ML models, and tooling baked into everything we ship. I have inserted a "
        "Background IP carve-out in the paragraph below \u2014 please review before "
        "we can agree to the \u201csolely in Client\u201d formulation. The carve-out "
        "is standard in every engagement we run.",
        "C0000007",
    )

    # Thread C: confidentiality (C8 root, C9 reply)
    c += comment_xml(
        8,
        "Jennifer Park",
        "2024-02-01T16:00:00Z",
        "JP",
        "Should this be longer? Our standard template uses five years for software "
        "development agreements, particularly where the contractor has access to our "
        "warehouse throughput and inventory data. I want to make sure we are protected "
        "if TechVenture later works with a direct competitor such as GlobalStock Inc.",
        "C0000008",
    )
    c += comment_xml(
        9,
        "Sarah Chen",
        "2024-02-02T09:30:00Z",
        "SC",
        "Jennifer \u2014 good instinct but getting to five years would be a hard "
        "negotiation. Two years was already a concession from their original ask of "
        "one year. We also have the non-compete in Schedule\u00a0C covering the top "
        "six retailers. My recommendation: hold at two years and only revisit if "
        "TechVenture pushes back on the non-compete scope.",
        "C0000009",
    )

    # Thread D: termination notice (C10 root, C11 reply, C12 reply)
    c += comment_xml(
        10,
        "David Okafor",
        "2024-02-03T08:45:00Z",
        "DO",
        "Thirty days is completely unworkable for TechVenture. We have three "
        "active subcontractors on this project (UI/UX, QA, DevOps), infrastructure "
        "commitments totalling USD\u00a034,000 per month, and a six-week minimum "
        "handover cycle for any client. We need ninety (90) days minimum or this "
        "deal does not work for us. I am flagging this as a blocker.",
        "C000000A",
    )
    c += comment_xml(
        11,
        "Sarah Chen",
        "2024-02-05T10:00:00Z",
        "SC",
        "David, we appreciate the transparency on the operational constraints. "
        "Acme is prepared to move to sixty (60) days as a compromise position. "
        "This is consistent with industry standard for contracts of this size "
        "and value and should provide adequate runway for an orderly wind-down. "
        "Please confirm if sixty days is acceptable so we can update the clause "
        "and progress to execution.",
        "C000000B",
    )
    c += comment_xml(
        12,
        "David Okafor",
        "2024-02-06T08:00:00Z",
        "DO",
        "Sixty days works for TechVenture. Agreed, subject to all remaining "
        "open points being resolved satisfactorily.",
        "C000000C",
    )

    # Thread E: governing law (C13 root, C14 reply, C15 reply)
    c += comment_xml(
        13,
        "Marcus Webb",
        "2024-02-10T14:00:00Z",
        "MW",
        "TechVenture is incorporated in Delaware and our external litigation counsel "
        "is based in San Francisco. New York law creates unnecessary cost and "
        "jurisdictional complexity for us in any dispute scenario. We would strongly "
        "prefer Delaware (our state of incorporation) or California as the governing "
        "law. Happy to discuss on a call.",
        "C000000D",
    )
    c += comment_xml(
        14,
        "Sarah Chen",
        "2024-02-11T09:00:00Z",
        "SC",
        "Acme\u2019s entire legal function runs on New York law. Our GC has final "
        "say on this and has said no. New York stays. This is not open for "
        "further discussion.",
        "C000000E",
    )
    c += comment_xml(
        15,
        "Marcus Webb",
        "2024-02-12T10:30:00Z",
        "MW",
        "Noted and accepted \u2014 TechVenture will accept New York governing law. "
        "However, we do require that any disputes be submitted to binding arbitration "
        "under AAA Commercial Arbitration Rules rather than court litigation. "
        "Arbitration is standard in all our commercial agreements and I trust this "
        "will not be an issue for Acme.",
        "C000000F",
    )

    # C16 standalone — fee schedule total (Jennifer, Jan 16, constructive)
    c += comment_xml(
        16,
        "Jennifer Park",
        "2024-01-16T09:00:00Z",
        "JP",
        "The USD\u00a0150,000 fixed fee is in line with the budget approved by "
        "Acme\u2019s Finance Committee on January 10, 2024 (ref FC-2024-003). "
        "Just confirming the milestone split (45/60/45) is consistent with "
        "the payment schedule discussed in the December 14 kickoff call. "
        "Please also confirm whether the support retainer of USD\u00a08,500/month "
        "is within the original scope or requires a separate PO.",
        "C0000010",
    )

    # Thread F: Phase 3 fee (C17 root, C18 reply, C19 reply)
    c += comment_xml(
        17,
        "David Okafor",
        "2024-01-18T14:00:00Z",
        "DO",
        "We agreed USD\u00a050,000 for Phase\u00a03 on the December 14 call. "
        "Reducing it to USD\u00a045,000 now is a unilateral change to a deal "
        "term that was already settled. The total fixed fee drops from "
        "USD\u00a0155,000 to USD\u00a0150,000. TechVenture will not absorb "
        "a USD\u00a05,000 haircut at this stage of negotiations.",
        "C0000011",
    )
    c += comment_xml(
        18,
        "Sarah Chen",
        "2024-01-19T09:30:00Z",
        "SC",
        "David \u2014 the December 14 call summary (attached) records the Phase\u00a03 "
        "fee as \u201cto be confirmed\u201d pending scope finalisation. The reduction "
        "reflects the descoping of the mobile app component agreed on January 8. "
        "The total fee of USD\u00a0150,000 is what Acme\u2019s board approved and "
        "we cannot go above that.",
        "C0000012",
    )
    c += comment_xml(
        19,
        "Marcus Webb",
        "2024-01-20T08:00:00Z",
        "MW",
        "Having reviewed the December 14 notes I believe there is a genuine "
        "misunderstanding on both sides. To move things forward: TechVenture will "
        "accept USD\u00a045,000 for Phase\u00a03 on the condition that Phase\u00a02 "
        "remains at USD\u00a060,000 and the support retainer is confirmed at "
        "USD\u00a08,500 per month. Please confirm and we can close this.",
        "C0000013",
    )

    # Thread G: warranty period (C20 root, C21 reply)
    c += comment_xml(
        20,
        "Sarah Chen",
        "2024-02-08T11:00:00Z",
        "SC",
        "Six months is dangerously short for an ERP integration of this complexity. "
        "Latent defects in warehouse management software can take six to nine months "
        "to surface in production. Acme\u2019s standard for custom software is "
        "twelve months and I see no basis to deviate here.",
        "C0000014",
    )
    c += comment_xml(
        21,
        "Marcus Webb",
        "2024-02-09T09:00:00Z",
        "MW",
        "Sarah, we understand the concern. Our reasoning: the UAT period under "
        "Phase\u00a03 already provides a structured acceptance window, and post "
        "go-live the support retainer (12 months, USD\u00a0102,000) covers bug-fixes "
        "and defects. Taken together the client has 12+ months of coverage \u2014 the "
        "warranty and retainer together achieve what a standalone 12-month warranty "
        "would. Six months for the formal warranty period is reasonable in that context.",
        "C0000015",
    )

    # Thread H: liability cap (C22 root, C23 reply, C24 reply)
    c += comment_xml(
        22,
        "Sarah Chen",
        "2024-02-15T10:00:00Z",
        "SC",
        "USD\u00a0500,000 is completely inadequate. The contract value alone is "
        "USD\u00a0252,000 (USD\u00a0150,000 fixed + USD\u00a0102,000 retainer). "
        "If the Software fails in production across three distribution centres we "
        "are looking at potential losses well above USD\u00a01,000,000. We need the "
        "cap at USD\u00a01,000,000 minimum.",
        "C0000016",
    )
    c += comment_xml(
        23,
        "Marcus Webb",
        "2024-02-16T09:15:00Z",
        "MW",
        "Sarah, a USD\u00a01,000,000 cap on a USD\u00a0150,000 fixed-fee engagement "
        "is a 6.7\u00d7 multiplier \u2014 that is not a commercially rational risk "
        "allocation for a software vendor. USD\u00a0500,000 equals 3.3\u00d7 the "
        "fixed fee and is at the upper bound of what TechVenture\u2019s professional "
        "indemnity insurance covers. We cannot go above USD\u00a0500,000.",
        "C0000017",
    )
    c += comment_xml(
        24,
        "David Okafor",
        "2024-02-17T08:30:00Z",
        "DO",
        "To add context: TechVenture carries USD\u00a01,000,000 in professional "
        "indemnity cover but half of that is already committed to two other active "
        "engagements. USD\u00a0500,000 is the hard ceiling from our insurer's "
        "perspective. Marcus is correct on this one.",
        "C0000018",
    )

    # C25 standalone — sign-off (Jennifer, Feb 22, constructive)
    c += comment_xml(
        25,
        "Jennifer Park",
        "2024-02-22T17:00:00Z",
        "JP",
        "From Acme\u2019s engineering side: all substantive points are now resolved "
        "to my satisfaction. The milestone dates in Exhibit\u00a0B align with our "
        "Q3 go-live target for the Newark pilot. Sarah \u2014 please confirm from "
        "legal and we can move to final execution. Well done to everyone on "
        "getting this over the line.",
        "C0000019",
    )

    comments_xml = wrap_comments(c)

    # ── commentsExtended ──────────────────────────────────────────────────────

    ex = ""
    ex += comment_ex("C0000001")  # C1 standalone
    ex += comment_ex("C0000002")  # C2 standalone
    ex += comment_ex("C0000003")  # Thread A root
    ex += comment_ex("C0000004", parent_para_id="C0000003")  # A reply
    ex += comment_ex("C0000005", parent_para_id="C0000003")  # A reply
    ex += comment_ex("C0000006")  # Thread B root
    ex += comment_ex("C0000007", parent_para_id="C0000006")  # B reply
    ex += comment_ex("C0000008")  # Thread C root
    ex += comment_ex("C0000009", parent_para_id="C0000008")  # C reply
    ex += comment_ex("C000000A")  # Thread D root
    ex += comment_ex("C000000B", parent_para_id="C000000A")  # D reply
    ex += comment_ex("C000000C", parent_para_id="C000000A")  # D reply
    ex += comment_ex("C000000D")  # Thread E root
    ex += comment_ex("C000000E", parent_para_id="C000000D")  # E reply
    ex += comment_ex("C000000F", parent_para_id="C000000D")  # E reply
    ex += comment_ex("C0000010")  # C16 standalone
    ex += comment_ex("C0000011")  # Thread F root
    ex += comment_ex("C0000012", parent_para_id="C0000011")  # F reply
    ex += comment_ex("C0000013", parent_para_id="C0000011")  # F reply
    ex += comment_ex("C0000014")  # Thread G root
    ex += comment_ex("C0000015", parent_para_id="C0000014")  # G reply
    ex += comment_ex("C0000016")  # Thread H root
    ex += comment_ex("C0000017", parent_para_id="C0000016")  # H reply
    ex += comment_ex("C0000018", parent_para_id="C0000016")  # H reply
    ex += comment_ex("C0000019")  # C25 standalone

    comments_extended_xml = wrap_comments_extended(ex)

    # ── commentsIds ───────────────────────────────────────────────────────────

    para_ids = [
        "C0000001",
        "C0000002",
        "C0000003",
        "C0000004",
        "C0000005",
        "C0000006",
        "C0000007",
        "C0000008",
        "C0000009",
        "C000000A",
        "C000000B",
        "C000000C",
        "C000000D",
        "C000000E",
        "C000000F",
        "C0000010",
        "C0000011",
        "C0000012",
        "C0000013",
        "C0000014",
        "C0000015",
        "C0000016",
        "C0000017",
        "C0000018",
        "C0000019",
    ]
    ids = "".join(comment_id_entry(pid, i) for i, pid in enumerate(para_ids, start=1))
    comments_ids_xml = wrap_comments_ids(ids)

    write_docx(
        os.path.join(OUTPUT_DIR, "services_agreement.docx"),
        document_xml,
        comments_xml,
        comments_extended_xml,
        comments_ids_xml,
    )


# ===========================================================================
# DOCUMENT 2: Mutual Non-Disclosure Agreement
# ===========================================================================
#
# Parties:    Meridian Capital Partners ("Meridian")  vs  Helix Genomics Inc. ("Helix")
# Context:    USD 5,000,000 Series A investment exploration
# Authors:    Priya Nair (Meridian counsel)  |  Tom Reilly (Helix counsel)
#             Alex Foster (Helix CEO)        |  Diane Wu (Meridian Partner)
# Date range: 2024-03-01 → 2024-04-10
#
# Track changes (9 redlines):
#   1/2:   CI definition narrowed: broad "any and all" → marked/written  (Tom, Mar 05)
#   3:     Permitted disclosure carve-out (auditors/advisors) inserted  (Tom, Mar 20)
#   4:     MOVE: compelled disclosure para moved from §2 to §7  (Tom, Mar 22)
#   5/6:   Survival period: "three (3)" → "one (1)" year  (Tom, Apr 02)
#   7:     Return period: "sixty (60) days" → "thirty (30) days"  (Priya, Apr 05)
#   8/9:   Non-solicitation: "twenty-four (24) months" → "twelve (12) months"  (Tom, Mar 28)
#
# Comments (15 total / 5 threads):
#   D1   Tom    / Mar 05 — CI definition too broad (root thread A, pushback)
#   D2   Priya  / Mar 06 — defends broad definition (reply A, firm)
#   D3   Tom    / Mar 07 — compromise proposal (reply A, constructive)
#   D4   Priya  / Mar 10 — standstill provision question (standalone)
#   D5   Alex   / Mar 15 — genomics pipeline coverage (standalone, informal)
#   D6   Tom    / Mar 20 — auditor/advisor carve-out (root thread B, constructive)
#   D7   Priya  / Mar 21 — accepts with conditions (reply B, diplomatic)
#   D8   Priya  / Apr 02 — survival period pushback (root thread C, blunt)
#   D9   Tom    / Apr 03 — rationale for 1 year (reply C, diplomatic)
#   D10  Priya  / Apr 04 — conditional acceptance (reply C)
#   D11  Alex   / Mar 12 — data room timing question (standalone, informal)
#   D12  Tom    / Mar 28 — non-solicitation too long (root thread D, firm)
#   D13  Priya  / Mar 29 — defends 24 months (reply D)
#   D14  Priya  / Apr 08 — return of materials timeline (root thread E)
#   D15  Diane  / Apr 10 — final sign-off from Meridian (reply E, constructive)
# ===========================================================================


def make_nda() -> None:

    body = ""

    # ── Title ─────────────────────────────────────────────────────────────────
    body += title_para("MUTUAL NON-DISCLOSURE AGREEMENT")
    body += blank()

    # ── Preamble ──────────────────────────────────────────────────────────────
    body += para(
        run(
            "This Mutual Non-Disclosure Agreement (\u201cAgreement\u201d) is entered "
            "into as of March 1, 2024 (the \u201cEffective Date\u201d), between "
            "Meridian Capital Partners LP, a Delaware limited partnership with its "
            "principal place of business at 200 Park Avenue, New York, NY 10166 "
            "(\u201cMeridian\u201d), and Helix Genomics Inc., a California corporation "
            "with its principal place of business at 4000 Shoreline Court, South "
            "San Francisco, CA 94080 (\u201cHelix\u201d). Meridian and Helix are "
            "exploring a potential Series\u00a0A equity investment of up to "
            "USD\u00a05,000,000 (the \u201cProposed Transaction\u201d) and wish to "
            "protect confidential information exchanged in that context. "
            "The Parties are hereinafter referred to collectively as the "
            "\u201cParties\u201d and individually as a \u201cParty.\u201d"
        ),
        pid="20000001",
    )
    body += blank()

    # ── §1 CONFIDENTIAL INFORMATION  [TC 1/2; Thread A D1/D2/D3] ─────────────
    body += heading("1.  CONFIDENTIAL INFORMATION", pid="20000002")
    body += para(
        run("\u201cConfidential Information\u201d means ")
        + del_(
            "any and all information, whether written, oral, electronic, or in "
            "any other form, that is disclosed by one party (the \u201cDisclosing "
            "Party\u201d) to the other (the \u201cReceiving Party\u201d)",
            "Tom Reilly",
            "2024-03-05T10:00:00Z",
            1,
        )
        + cstart(1)
        + ins(
            "information disclosed by the Disclosing Party that is either: (a) "
            "in writing and marked \u201cConfidential\u201d or \u201cProprietary\u201d "
            "at the time of disclosure; or (b) if disclosed orally or visually, "
            "identified as confidential at the time of disclosure and confirmed in "
            "a written summary delivered to the Receiving Party within five (5) "
            "business days thereafter",
            "Tom Reilly",
            "2024-03-05T10:00:00Z",
            2,
        )
        + cend(1)
        + crefs(1, 2, 3)
        + run(
            " and that relates to the Disclosing Party\u2019s business, "
            "technology, financial information, genomics pipeline data, "
            "clinical trial data, intellectual property, or strategic plans "
            "(\u201cConfidential Information\u201d). Confidential Information "
            "does not include information that: (i)\u00a0is or becomes publicly "
            "known through no fault of the Receiving Party; (ii)\u00a0was already "
            "known to the Receiving Party prior to disclosure, as evidenced by "
            "written records predating disclosure; (iii)\u00a0is independently "
            "developed by the Receiving Party without reference to the Confidential "
            "Information; or (iv)\u00a0is received from a third party without "
            "restriction and without breach of any obligation of confidentiality."
        ),
        pid="20000003",
    )
    body += blank()

    # ── §2 OBLIGATIONS  [TC 3 inserted; MOVE from here: compelled disclosure; D6/D7] ──
    body += heading("2.  OBLIGATIONS OF RECEIVING PARTY", pid="20000004")
    body += para(
        run(
            "The Receiving Party shall: (a)\u00a0hold all Confidential Information "
            "in strict confidence, using at least the same degree of care it uses "
            "for its own confidential information of like nature, but in no event "
            "less than reasonable care; (b)\u00a0not disclose Confidential Information "
            "to any third party without the prior written consent of the Disclosing "
            "Party; (c)\u00a0use Confidential Information solely for evaluating "
            "and consummating the Proposed Transaction; and (d)\u00a0promptly notify "
            "the Disclosing Party in writing upon discovery of any unauthorised use "
            "or disclosure of Confidential Information."
        ),
        pid="20000005",
    )
    # Permitted disclosure carve-out inserted by Tom [TC 3]
    body += para(
        cstart(6)
        + ins(
            "Notwithstanding clause 2(b), the Receiving Party may disclose "
            "Confidential Information, on a strict need-to-know basis, to its "
            "directors, officers, employees, legal advisors, accountants, and "
            "financing sources (\u201cAuthorised Recipients\u201d) who are bound "
            "by confidentiality obligations no less restrictive than those set out "
            "in this Agreement. The Receiving Party shall be responsible for any "
            "breach of this Agreement by its Authorised Recipients.",
            "Tom Reilly",
            "2024-03-20T14:00:00Z",
            3,
        )
        + cend(6)
        + crefs(6, 7),
        pid="20000006",
    )
    # Compelled disclosure para — MOVED FROM HERE [MOVE: rid=4]
    body += move_from_block(
        para(
            run(
                "If the Receiving Party is required by applicable law, regulation, "
                "or court order to disclose any Confidential Information, it shall: "
                "(i)\u00a0give the Disclosing Party prompt written notice (where "
                "legally permissible) to allow the Disclosing Party to seek a "
                "protective order or other appropriate relief; (ii)\u00a0cooperate "
                "reasonably with the Disclosing Party in connection with any such "
                "effort; and (iii)\u00a0disclose only that portion of the Confidential "
                "Information that is legally required to be disclosed."
            ),
            pid="20000007",
        ),
        "Tom Reilly",
        "2024-03-22T09:00:00Z",
        4,
    )
    body += blank()

    # ── §3 RETURN OF INFORMATION  [TC 7 on period; D14/D15] ──────────────────
    body += heading("3.  RETURN OF INFORMATION", pid="20000008")
    body += para(
        cstart(14)
        + run(
            "Upon the written request of the Disclosing Party or upon termination "
            "of this Agreement, the Receiving Party shall promptly, and in any "
            "event within "
        )
        + del_("sixty (60) days", "Priya Nair", "2024-04-05T10:00:00Z", 7)
        + ins("thirty (30) days", "Priya Nair", "2024-04-05T10:00:00Z", 8)
        + run(
            ", return or certifiably destroy all Confidential Information "
            "(including all copies, notes, analyses, and derivative works) in "
            "its possession or control. Upon request, the Receiving Party shall "
            "provide a written certification signed by an authorised officer "
            "confirming such return or destruction. Notwithstanding the foregoing, "
            "the Receiving Party may retain one archival copy of Confidential "
            "Information to the extent required by applicable law or its bona fide "
            "document retention policies, provided such copy remains subject to "
            "the confidentiality obligations of this Agreement."
        )
        + cend(14)
        + crefs(14, 15),
        pid="20000009",
    )
    body += blank()

    # ── §4 NON-SOLICITATION  [TC 8/9 on period; D12/D13] ─────────────────────
    body += heading("4.  NON-SOLICITATION", pid="2000000A")
    body += para(
        run("During the term of this Agreement and for a period of ")
        + del_("twenty-four (24) months", "Tom Reilly", "2024-03-28T11:00:00Z", 9)
        + cstart(12)
        + ins("twelve (12) months", "Tom Reilly", "2024-03-28T11:00:00Z", 10)
        + cend(12)
        + crefs(12, 13)
        + run(
            " following termination, neither party shall, directly or indirectly, "
            "solicit or recruit for employment any person who is then employed by "
            "the other party and with whom the first party had material contact "
            "in connection with the Proposed Transaction, without the prior written "
            "consent of the other party. This restriction shall not apply to general "
            "solicitations (such as job postings or advertisements) not specifically "
            "targeted at employees of the other party."
        ),
        pid="2000000B",
    )
    body += blank()

    # ── §5 TERM  [D4 standalone on standstill; TC 5/6; Thread C D8/D9/D10] ───
    body += heading("5.  TERM", pid="2000000C")
    body += para(
        cstart(4)
        + run("This Agreement shall remain in effect for two (2) years")
        + cend(4)
        + cref(4)
        + run(
            " from the Effective Date, unless earlier terminated by either Party "
            "on thirty (30) days\u2019 prior written notice. The obligations of "
            "confidentiality shall survive termination of this Agreement for a "
            "further period of "
        )
        + del_("three (3)", "Tom Reilly", "2024-04-02T09:00:00Z", 5)
        + cstart(8)
        + ins("one (1)", "Tom Reilly", "2024-04-02T09:00:00Z", 6)
        + cend(8)
        + crefs(8, 9, 10)
        + run(" year following such termination."),
        pid="2000000D",
    )
    body += blank()

    # ── §6 REMEDIES ───────────────────────────────────────────────────────────
    body += heading("6.  REMEDIES", pid="2000000E")
    body += para(
        run(
            "The Parties acknowledge that a breach of this Agreement would cause "
            "irreparable harm for which monetary damages would be an inadequate "
            "remedy and that the precise amount of such damages would be difficult "
            "to ascertain. Accordingly, each Party shall be entitled to seek "
            "equitable relief, including injunction and specific performance, "
            "without the requirement to post a bond or other security, in addition "
            "to all other remedies available at law or in equity. In the event of "
            "a breach involving the genomics pipeline data or clinical trial data "
            "of Helix, Helix shall also be entitled to seek an accounting of profits "
            "derived from unauthorised use of such data."
        ),
        pid="2000000F",
    )
    body += blank()

    # ── §7 COMPELLED DISCLOSURE — MOVED TO HERE ────────────────────────────────
    body += heading("7.  COMPELLED DISCLOSURE", pid="20000010")
    body += move_to_block(
        para(
            run(
                "If the Receiving Party is required by applicable law, regulation, "
                "or court order to disclose any Confidential Information, it shall: "
                "(i)\u00a0give the Disclosing Party prompt written notice (where "
                "legally permissible) to allow the Disclosing Party to seek a "
                "protective order or other appropriate relief; (ii)\u00a0cooperate "
                "reasonably with the Disclosing Party in connection with any such "
                "effort; and (iii)\u00a0disclose only that portion of the Confidential "
                "Information that is legally required to be disclosed."
            ),
            pid="20000011",
        ),
        "Tom Reilly",
        "2024-03-22T09:00:00Z",
        4,
    )
    body += blank()

    # ── §8 DISPUTE RESOLUTION  [D5 standalone] ────────────────────────────────
    body += heading("8.  DISPUTE RESOLUTION", pid="20000012")
    body += para(
        cstart(5)
        + run(
            "Any dispute arising out of or relating to this Agreement that cannot "
            "be resolved by good-faith negotiation between senior representatives "
            "of the Parties within thirty (30) days of written notice of the "
            "dispute shall be submitted to binding arbitration administered by "
            "JAMS under its Streamlined Arbitration Rules and Procedures. "
            "The arbitration shall be conducted by a single arbitrator and shall "
            "take place in New York, New York. The decision of the arbitrator "
            "shall be final and binding upon the Parties and may be entered as a "
            "judgment in any court of competent jurisdiction. Each Party shall "
            "bear its own costs of the arbitration, with the arbitrator\u2019s "
            "fees split equally."
        )
        + cend(5)
        + cref(5),
        pid="20000013",
    )
    body += blank()

    # ── §9 GENERAL ─────────────────────────────────────────────────────────────
    body += heading("9.  GENERAL", pid="20000014")
    body += para(
        run(
            "This Agreement constitutes the entire agreement between the Parties "
            "relating to the subject matter hereof and supersedes all prior "
            "discussions and agreements with respect thereto. This Agreement may "
            "be amended only by a written instrument signed by authorised "
            "representatives of both Parties. No failure or delay by a Party in "
            "exercising any right hereunder shall constitute a waiver of that right. "
            "This Agreement shall be governed by the laws of the State of Delaware "
            "without regard to its conflict-of-laws principles. If any provision "
            "is unenforceable, that provision shall be modified to the minimum "
            "extent necessary to make it enforceable and the remaining provisions "
            "shall continue in full force. This Agreement may be executed in "
            "counterparts, each of which shall constitute an original, and "
            "electronic signatures shall be deemed valid."
        ),
        pid="20000015",
    )

    document_xml = wrap_document(body)

    # ── Comments ──────────────────────────────────────────────────────────────

    c = ""

    # Thread A: CI definition (D1 root, D2 reply, D3 reply)
    c += comment_xml(
        1,
        "Tom Reilly",
        "2024-03-05T10:00:00Z",
        "TR",
        "This definition is far too broad. \u201cAny and all information in any "
        "form\u201d would sweep in casual hallway conversations, publicly available "
        "press releases, and information Helix has no intent to protect. From a "
        "practical standpoint it creates uncertainty about what is actually covered. "
        "We need either a marking requirement or a reasonable specificity threshold. "
        "I\u2019ve proposed tighter language in tracked changes.",
        "D0000001",
    )
    c += comment_xml(
        2,
        "Priya Nair",
        "2024-03-06T09:00:00Z",
        "PN",
        "Tom, the broad definition is intentional and standard in Meridian\u2019s "
        "investment NDA template. We share materials verbally in data room sessions "
        "and on analyst calls \u2014 a marking requirement would leave us exposed "
        "for anything disclosed orally. Meridian\u2019s position is to revert to "
        "the original language. We can discuss on a call if needed.",
        "D0000002",
    )
    c += comment_xml(
        3,
        "Tom Reilly",
        "2024-03-07T08:30:00Z",
        "TR",
        "I understand the concern about oral disclosures. Proposed compromise: keep "
        "the written marking requirement but add a catch-all covering oral disclosures "
        "that are confirmed in writing within 5 business days. This is the NVCA "
        "standard formulation and gives you the protection you need while providing "
        "Helix the certainty our board requires before we open the technical data room. "
        "I\u2019ve drafted this in the tracked changes above.",
        "D0000003",
    )

    # D4 standalone — standstill question (Priya, Mar 10)
    c += comment_xml(
        4,
        "Priya Nair",
        "2024-03-10T11:00:00Z",
        "PN",
        "Should we include a standstill provision preventing Helix from approaching "
        "Meridian\u2019s portfolio companies (list attached as Annex\u00a01) during "
        "the two-year term? Given that Meridian has portfolio companies operating "
        "in adjacent genomics and diagnostics spaces, this is something we typically "
        "require in Series\u00a0A investment NDAs. Please advise.",
        "D0000004",
    )

    # D5 standalone — Alex informal (Mar 15)
    c += comment_xml(
        5,
        "Alex Foster",
        "2024-03-15T17:30:00Z",
        "AF",
        "Hi all \u2014 just looped in by Tom. Quick question: does this NDA cover "
        "our CRISPR-Cas12 pipeline data specifically? That\u2019s the crown jewel "
        "and the basis for the USD\u00a05M valuation. Before we start sharing the "
        "technical data room (Phase\u00a02 data, IND filings, partnership term "
        "sheets) I want to be absolutely sure it\u2019s airtight. Thanks \u2014 Alex",
        "D0000005",
    )

    # Thread B: auditor carve-out (D6 root, D7 reply)
    c += comment_xml(
        6,
        "Tom Reilly",
        "2024-03-20T14:00:00Z",
        "TR",
        "We need an explicit carve-out permitting disclosure to our legal counsel, "
        "auditors (Deloitte), and financing sources (our bridge lender, Pacific "
        "Ventures). Without this, Helix\u2019s CFO cannot involve the accountants "
        "in due diligence financial review, which is a non-starter for the Series\u00a0A "
        "process. This is a standard carve-out in all investment NDAs.",
        "D0000006",
    )
    c += comment_xml(
        7,
        "Priya Nair",
        "2024-03-21T09:00:00Z",
        "PN",
        "Meridian can accept a carve-out for Authorised Recipients on a need-to-know "
        "basis, provided such persons are bound by confidentiality obligations "
        "materially no less restrictive than this Agreement and that Helix remains "
        "responsible for any breach by its advisors. I\u2019ve noted the conditions "
        "in the revised clause. Please confirm the language is workable for your team.",
        "D0000007",
    )

    # Thread C: survival period (D8 root, D9 reply, D10 reply)
    c += comment_xml(
        8,
        "Priya Nair",
        "2024-04-02T09:00:00Z",
        "PN",
        "Three years survival is standard for investment NDAs of this type and was "
        "in Meridian\u2019s template. One year is wholly insufficient for the "
        "information we are sharing here \u2014 particularly the pipeline valuation "
        "models and the Phase\u00a02 clinical data which will not be public for at "
        "least 18\u201324 months. Please revert to three years.",
        "D0000008",
    )
    c += comment_xml(
        9,
        "Tom Reilly",
        "2024-04-03T10:00:00Z",
        "TR",
        "Priya, Helix is a growth-stage company and our Series\u00a0A investors "
        "have flagged long-tail confidentiality obligations as a diligence concern. "
        "The genomics landscape moves rapidly \u2014 information that is commercially "
        "sensitive today (e.g., the Cas12 target identification data) is likely to "
        "be published or patented within 18 months. One year post-term reflects "
        "that reality. We genuinely cannot accept three years.",
        "D0000009",
    )
    c += comment_xml(
        10,
        "Priya Nair",
        "2024-04-04T08:00:00Z",
        "PN",
        "Understood the rationale. Meridian can meet at two (2) years provided "
        "the pipeline valuation models and any Phase\u00a02 clinical data are "
        "explicitly designated as a longer-protected category (minimum three years) "
        "in an annex to this Agreement. Happy to draft the annex language. "
        "Let\u2019s discuss on the Thursday call.",
        "D000000A",
    )

    # D11 standalone — Alex data room timing (Mar 12)
    c += comment_xml(
        11,
        "Alex Foster",
        "2024-03-12T15:00:00Z",
        "AF",
        "Priya / Tom \u2014 quick practical question: once both parties sign "
        "this NDA, how long before we can open the Phase\u00a02 data room? "
        "Helix\u2019s board has given us a 90-day exclusivity window starting "
        "March 1 and we\u2019re already 12 days in. The data room needs to be "
        "open by March 25 at the latest for the timeline to work. "
        "Can we expedite signature?",
        "D000000B",
    )

    # Thread D: non-solicitation (D12 root, D13 reply)
    c += comment_xml(
        12,
        "Tom Reilly",
        "2024-03-28T11:00:00Z",
        "TR",
        "Twenty-four months non-solicitation is excessive and unusual in an NDA "
        "of this type. For a Series\u00a0A process involving fewer than five Helix "
        "employees, a 24-month restriction on recruiting would effectively freeze "
        "Helix\u2019s ability to hire senior talent \u2014 several of our key "
        "scientific staff are active in the Meridian portfolio network. We propose "
        "twelve (12) months maximum.",
        "D000000C",
    )
    c += comment_xml(
        13,
        "Priya Nair",
        "2024-03-29T09:00:00Z",
        "PN",
        "Meridian\u2019s standard for investment NDAs is 24 months, reflecting the "
        "length of the post-investment relationship and the depth of access our "
        "team will have to Helix personnel during due diligence. That said, we "
        "recognise this is longer than market for a pre-investment NDA. Meridian "
        "will consider eighteen (18) months as a compromise, but cannot go below "
        "that given our portfolio conflict concerns.",
        "D000000D",
    )

    # Thread E: return of materials (D14 root, D15 reply)
    c += comment_xml(
        14,
        "Priya Nair",
        "2024-04-08T10:00:00Z",
        "PN",
        "On Section\u00a03: sixty days is too long for return of materials in an "
        "investment context where the deal may not proceed. If the Proposed "
        "Transaction is not consummated, Meridian needs its financial models and "
        "board presentation materials back within thirty days. I\u2019ve updated "
        "the clause \u2014 please confirm this is workable.",
        "D000000E",
    )
    c += comment_xml(
        15,
        "Diane Wu",
        "2024-04-10T14:00:00Z",
        "DW",
        "Confirmed from Meridian\u2019s side \u2014 thirty days is workable and "
        "consistent with our operational capacity. All open issues on my review "
        "list are now resolved. Priya, please coordinate with Tom on final "
        "execution logistics. We are targeting signature by April 15 to preserve "
        "the Helix exclusivity window.",
        "D000000F",
    )

    comments_xml = wrap_comments(c)

    # ── commentsExtended ──────────────────────────────────────────────────────

    ex = ""
    ex += comment_ex("D0000001")  # Thread A root
    ex += comment_ex("D0000002", parent_para_id="D0000001")  # A reply
    ex += comment_ex("D0000003", parent_para_id="D0000001")  # A reply
    ex += comment_ex("D0000004")  # D4 standalone
    ex += comment_ex("D0000005")  # D5 standalone
    ex += comment_ex("D0000006")  # Thread B root
    ex += comment_ex("D0000007", parent_para_id="D0000006")  # B reply
    ex += comment_ex("D0000008")  # Thread C root
    ex += comment_ex("D0000009", parent_para_id="D0000008")  # C reply
    ex += comment_ex("D000000A", parent_para_id="D0000008")  # C reply
    ex += comment_ex("D000000B")  # D11 standalone
    ex += comment_ex("D000000C")  # Thread D root
    ex += comment_ex("D000000D", parent_para_id="D000000C")  # D reply
    ex += comment_ex("D000000E")  # Thread E root
    ex += comment_ex("D000000F", parent_para_id="D000000E")  # E reply

    comments_extended_xml = wrap_comments_extended(ex)

    # ── commentsIds ───────────────────────────────────────────────────────────

    para_ids = [
        "D0000001",
        "D0000002",
        "D0000003",
        "D0000004",
        "D0000005",
        "D0000006",
        "D0000007",
        "D0000008",
        "D0000009",
        "D000000A",
        "D000000B",
        "D000000C",
        "D000000D",
        "D000000E",
        "D000000F",
    ]
    ids = "".join(comment_id_entry(pid, i) for i, pid in enumerate(para_ids, start=1))
    comments_ids_xml = wrap_comments_ids(ids)

    write_docx(
        os.path.join(OUTPUT_DIR, "nda.docx"),
        document_xml,
        comments_xml,
        comments_extended_xml,
        comments_ids_xml,
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating test documents...")
    make_services_agreement()
    make_nda()
    print("Done.")
