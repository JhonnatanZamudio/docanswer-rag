"""Generate the synthetic document corpus for the DocAnswer demo.

Every document belongs to "Kestrel Ridge Logistics", a FICTIONAL company. All names,
numbers and procedures are invented for demonstration purposes.

Each page is declared explicitly so that the expected (doc, page) for every evaluation
question is deterministic. Writes PDFs plus `manifest.json` (title, access level, language).

Usage: python scripts/make_corpus.py [--out data/docs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab import rl_config
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

rl_config.invariant = 1  # byte-for-byte reproducible PDFs (no timestamps / random IDs)

COMPANY = "Kestrel Ridge Logistics"

# doc_id -> metadata + pages; each page is a list of (heading, [paragraphs]).
DOCS: dict[str, dict] = {
    "employee-handbook": {
        "title": "Employee Handbook",
        "short": "Employee Handbook",
        "access": "employees",
        "lang": "en",
        "pages": [
            [
                (
                    "1. Welcome and scope",
                    [
                        "This handbook describes the everyday rules that apply to all permanent employees of "
                        "Kestrel Ridge Logistics in Colombia. Contractors are covered by their own agreements.",
                        "Where this handbook and a specific policy disagree, the specific policy prevails. "
                        "Questions about this handbook should be sent to the People team through PeopleHub.",
                    ],
                ),
                (
                    "2. Working hours",
                    [
                        "The standard working week is 42 hours, Monday to Friday. Core hours are 9:00 to 16:00, "
                        "during which every employee is expected to be available for meetings.",
                        "Overtime must be approved in advance by your manager and is registered in PeopleHub "
                        "before the end of the same week.",
                    ],
                ),
                (
                    "3. Probation period",
                    [
                        "New employees have a probation period of 60 calendar days. During probation either party "
                        "may end the contract without notice. Remote work is not available during probation.",
                    ],
                ),
            ],
            [
                (
                    "4. Sick leave",
                    [
                        "If you are sick, notify your manager before 10:00 on the first day of absence. A medical "
                        "certificate is required when the absence lasts more than 2 consecutive days.",
                        "Certificates are uploaded to PeopleHub within 3 business days of returning to work.",
                    ],
                ),
                (
                    "5. Family and personal leave",
                    [
                        "Maternity leave is 18 weeks and paternity leave is 2 weeks, as established by Colombian law. "
                        "Employees may request up to 4 additional unpaid weeks immediately after parental leave.",
                        "Bereavement leave is 5 business days for the death of a spouse, partner, parent, child or "
                        "sibling. Employees who get married receive 3 paid days of marriage leave.",
                    ],
                ),
            ],
            [
                (
                    "6. Remote and hybrid work",
                    [
                        "After completing probation, employees in eligible roles may work remotely up to 3 days per "
                        "week. Warehouse, fleet and reception roles are not eligible because they require "
                        "presence on site.",
                        "Remote employees receive a one-time home office equipment stipend of COP 1,200,000 and a "
                        "monthly internet allowance of COP 120,000, paid with the regular payroll.",
                        "Working from another country is limited to 20 days per calendar year and requires written "
                        "approval from the People team at least 15 days before travelling.",
                    ],
                ),
            ],
            [
                (
                    "7. Benefits",
                    [
                        "The company pays 70% of the premium of the complementary health plan for the employee. "
                        "Dependants can be added at the employee's own cost.",
                        "Each employee has an annual learning budget of USD 600 for courses, certifications or books, "
                        "approved by the manager in PeopleHub.",
                    ],
                ),
                (
                    "8. Performance reviews",
                    [
                        "Performance reviews take place twice a year, in June and in December. Each review combines a "
                        "self-assessment, manager feedback and agreed goals for the next six months.",
                    ],
                ),
                (
                    "9. Code of conduct",
                    [
                        "Gifts or invitations from suppliers or clients worth more than USD 50 must be declared to "
                        "the Compliance team within 5 business days.",
                        "Concerns about unethical behaviour can be reported anonymously to "
                        "ethics@kestrelridge.example.",
                    ],
                ),
            ],
        ],
    },
    "travel-expense-policy": {
        "title": "Travel & Expense Policy",
        "short": "Travel Policy",
        "access": "employees",
        "lang": "en",
        "pages": [
            [
                (
                    "1. Purpose",
                    [
                        "This policy defines how business travel is requested, booked and reimbursed. It applies to "
                        "every employee who travels on behalf of Kestrel Ridge Logistics.",
                    ],
                ),
                (
                    "2. Trip approval",
                    [
                        "All trips must be requested in TravelDesk at least 10 business days before departure. "
                        "Domestic trips are approved by the direct manager.",
                        "International trips additionally require approval from the area director. Trips booked "
                        "without approval are not reimbursed.",
                    ],
                ),
            ],
            [
                (
                    "3. Flights",
                    [
                        "Economy class is mandatory for flights shorter than 6 hours. Premium economy is allowed for "
                        "flights of 6 hours or more.",
                        "Business class is only allowed for vice presidents on flights longer than 10 hours.",
                    ],
                ),
                (
                    "4. Hotels",
                    [
                        "The nightly hotel cap is USD 180 in the United States and Europe and USD 120 in other Latin "
                        "American countries.",
                        "In Colombia the cap is COP 350,000 per night in Bogota and Medellin and COP 280,000 per night "
                        "in other cities. Bookings above the cap need written justification.",
                    ],
                ),
            ],
            [
                (
                    "5. Meals and per diem",
                    [
                        "The meal per diem is USD 65 per day for international trips and COP 140,000 per day for "
                        "domestic trips. Alcoholic drinks are never reimbursable.",
                        "Client entertainment above USD 200 per event requires pre-approval from the area director.",
                    ],
                ),
                (
                    "6. Ground transportation",
                    [
                        "Taxis and ride-hailing services are allowed. Employees who use their own car are reimbursed "
                        "COP 1,100 per kilometre, plus tolls and parking.",
                    ],
                ),
            ],
            [
                (
                    "7. Expense reports and reimbursement",
                    [
                        "Expense reports must be submitted in TravelDesk within 15 calendar days after the trip ends. "
                        "Approved reports are paid in the next payroll cycle.",
                        "Receipts are required for any expense above USD 25 or COP 100,000. A lost receipt form may "
                        "be used at most 2 times per year.",
                    ],
                ),
                (
                    "8. Corporate card",
                    [
                        "Employees who travel more than 4 times per year may request a corporate card. Personal "
                        "expenses on the corporate card must be repaid within 30 days.",
                    ],
                ),
            ],
        ],
    },
    "it-security-policy": {
        "title": "IT Security Policy",
        "short": "IT Security Policy",
        "access": "public",
        "lang": "en",
        "pages": [
            [
                (
                    "1. Scope",
                    [
                        "This policy applies to employees, contractors and visitors who use any Kestrel Ridge "
                        "Logistics system, network or device.",
                    ],
                ),
                (
                    "2. Passwords and MFA",
                    [
                        "Passwords must have at least 14 characters; long passphrases are recommended. Passwords are "
                        "changed only when there is a suspicion of compromise, not on a fixed schedule.",
                        "Multi-factor authentication is mandatory for email, VPN and the PeopleHub HR portal. "
                        "Passwords must be stored in the company password manager.",
                    ],
                ),
            ],
            [
                (
                    "3. Devices",
                    [
                        "Company laptops use full-disk encryption and lock the screen after 5 minutes of inactivity. "
                        "USB storage devices are blocked by default.",
                        "Personal phones may access company email only after enrolment in the mobile device "
                        "management system.",
                    ],
                ),
                (
                    "4. Phishing",
                    [
                        "Suspicious emails must be reported with the Report Phishing button in the mail client. "
                        "The security team runs phishing simulations every quarter.",
                    ],
                ),
            ],
            [
                (
                    "5. Incident reporting",
                    [
                        "A lost or stolen laptop or phone must be reported within 1 hour to the security desk at "
                        "extension 4400 or security@kestrelridge.example.",
                    ],
                ),
                (
                    "6. Guest Wi-Fi",
                    [
                        "Visitors use the KR-Guest network. Reception issues a voucher that is valid for 24 hours.",
                    ],
                ),
                (
                    "7. Data classification",
                    [
                        "Information is classified as Public, Internal, Confidential or Restricted. Restricted data, "
                        "such as payroll records, may never be sent to personal email accounts.",
                    ],
                ),
            ],
        ],
    },
    "onboarding-guide": {
        "title": "Onboarding Guide",
        "short": "Onboarding Guide",
        "access": "public",
        "lang": "en",
        "pages": [
            [
                (
                    "1. Your first day",
                    [
                        "Please arrive at 8:30 at the reception desk on floor 7 of Torre Kestrel. Bring your national "
                        "ID, your signed contract and a bank certificate so payroll can be set up.",
                        "Your manager will meet you at reception and walk you through the office.",
                    ],
                ),
            ],
            [
                (
                    "2. Your first week",
                    [
                        "Your laptop is handed over on the first day and your accounts are activated within 24 hours.",
                        "Every new employee is assigned a buddy for the first 90 days to answer day-to-day questions.",
                        "The security awareness training must be completed within the first 7 days and the harassment "
                        "prevention course within the first 30 days.",
                    ],
                ),
            ],
            [
                (
                    "3. Your first 90 days",
                    [
                        "You will have check-ins with your manager at 30, 60 and 90 days. The probation review "
                        "takes place on day 55.",
                    ],
                ),
                (
                    "4. Payroll",
                    [
                        "Salaries are paid monthly on the last business day of the month. Changes received after the "
                        "20th of the month are applied in the following payroll.",
                    ],
                ),
            ],
        ],
    },
    "politica-vacaciones": {
        "title": "Política de Vacaciones",
        "short": "Política de Vacaciones",
        "access": "employees",
        "lang": "es",
        "pages": [
            [
                (
                    "1. Objeto",
                    [
                        "Esta política regula el disfrute de las vacaciones de los empleados de Kestrel Ridge "
                        "Logistics en Colombia.",
                    ],
                ),
                (
                    "2. Derecho a vacaciones",
                    [
                        "Cada empleado tiene derecho a 15 días hábiles de vacaciones remuneradas por cada año de "
                        "servicio, de acuerdo con la ley colombiana.",
                        "Como beneficio adicional, la empresa otorga 1 día hábil extra por cada 5 años de antigüedad, "
                        "con un máximo de 3 días extra por año.",
                    ],
                ),
            ],
            [
                (
                    "3. Solicitud y aprobación",
                    [
                        "Las vacaciones se solicitan en el portal PeopleHub con mínimo 15 días calendario de "
                        "anticipación. El jefe inmediato debe responder la solicitud en un plazo de 5 días hábiles.",
                        "Al menos una vez al año el empleado debe tomar un periodo continuo de mínimo 6 días hábiles.",
                    ],
                ),
                (
                    "4. Acumulación y compensación",
                    [
                        "Se pueden acumular como máximo 2 periodos de vacaciones. Hasta el 50% de los días puede "
                        "compensarse en dinero, siempre por acuerdo escrito entre el empleado y la empresa.",
                    ],
                ),
            ],
            [
                (
                    "5. Temporada de cierre",
                    [
                        "Para los equipos de operaciones no se aprueban vacaciones entre el 15 de noviembre y el 10 de "
                        "enero, porque es la temporada alta de la operación logística.",
                    ],
                ),
                (
                    "6. Día de cumpleaños",
                    [
                        "Cada empleado recibe un día libre remunerado en la semana de su cumpleaños.",
                    ],
                ),
                (
                    "7. Interrupción de vacaciones",
                    [
                        "Si la empresa interrumpe las vacaciones por una necesidad urgente, los días no disfrutados se "
                        "reprograman de común acuerdo dentro de los 3 meses siguientes.",
                    ],
                ),
            ],
        ],
    },
    "manager-compensation-guidelines": {
        "title": "Manager Compensation Guidelines",
        "short": "Compensation Guidelines",
        "access": "managers",
        "lang": "en",
        "pages": [
            [
                (
                    "1. Purpose and confidentiality",
                    [
                        "These guidelines are confidential and intended only for people managers. They must not be "
                        "shared with individual contributors.",
                    ],
                ),
                (
                    "2. Salary bands and merit increases",
                    [
                        "Salary band L1 ranges from COP 3,200,000 to COP 4,500,000 per month and band L3 ranges from "
                        "COP 7,000,000 to COP 9,800,000 per month.",
                        "The merit increase budget for 2026 is 6.5% of the team payroll. An individual increase above "
                        "12% requires approval from the VP of People.",
                    ],
                ),
            ],
            [
                (
                    "3. Annual bonus",
                    [
                        "The annual bonus target is 8% of base salary. The payout multiplier ranges from 0% to 150% "
                        "depending on the performance rating, and bonuses are paid in March.",
                    ],
                ),
                (
                    "4. Promotions",
                    [
                        "Promotions are decided in two windows per year, in April and in October, after the "
                        "calibration committee. A promotion comes with a salary increase between 8% and 15%.",
                    ],
                ),
            ],
            [
                (
                    "5. Retention",
                    [
                        "Counter-offers always require approval from the Chief People Officer. A retention bonus can "
                        "be at most 20% of the annual salary and carries a 12-month clawback.",
                    ],
                ),
                (
                    "6. Terminations",
                    [
                        "Managers must consult the HR legal team before communicating any termination.",
                    ],
                ),
            ],
        ],
    },
}

ACCESS_LABEL = {"public": "PUBLIC", "employees": "INTERNAL - employees", "managers": "CONFIDENTIAL - managers"}


def build_pdf(doc_id: str, spec: dict, out_dir: Path) -> Path:
    path = out_dir / f"{doc_id}.pdf"
    styles = getSampleStyleSheet()
    footer = f"{COMPANY} (fictional) - {spec['title']} - v2026.1"

    def decorate(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, ACCESS_LABEL[spec["access"]])
        canvas.drawString(2 * cm, 1.2 * cm, f"{footer} - Page {doc.page}")
        canvas.restoreState()

    story: list = [
        Paragraph(spec["title"], styles["Title"]),
        Paragraph(f"{COMPANY} - fictional company for demonstration purposes", styles["Italic"]),
        Spacer(1, 0.6 * cm),
    ]
    for i, page in enumerate(spec["pages"]):
        if i > 0:
            story.append(PageBreak())
        for heading, paragraphs in page:
            story.append(Paragraph(heading, styles["Heading2"]))
            story.extend(Paragraph(p, styles["BodyText"]) for p in paragraphs)
    SimpleDocTemplate(
        str(path), pagesize=A4, title=spec["title"], author=COMPANY, topMargin=2 * cm, bottomMargin=2 * cm
    ).build(story, onFirstPage=decorate, onLaterPages=decorate)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/docs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for doc_id, spec in DOCS.items():
        build_pdf(doc_id, spec, args.out)
        manifest[f"{doc_id}.pdf"] = {k: spec[k] for k in ("title", "short", "access", "lang")}
        print(f"wrote {args.out / (doc_id + '.pdf')} ({len(spec['pages'])} pages, {spec['access']})")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
