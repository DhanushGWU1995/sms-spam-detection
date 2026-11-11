#!/usr/bin/env python3
"""Seed a curated official-guidance knowledge base (public pages + provenance)."""

from __future__ import annotations

from datetime import date

from backend.app.core.config import get_settings
from backend.app.core.s3_store import get_store


DOCS = [
    {
        "doc_id": "ftc_imposter_scams",
        "agency": "FTC",
        "title": "Imposter scams",
        "source_url": "https://consumer.ftc.gov/articles/imposter-scams",
        "topic": "government_impersonation",
        "text": (
            "Imposter scams happen when someone pretends to be a trusted person or organization. "
            "Scammers may claim to be from the government, a well-known company, or a relative in trouble. "
            "They often create urgency and ask you to pay with gift cards, wire transfers, or cryptocurrency. "
            "Do not send money or personal information based on an unexpected call, text, or email. "
            "Verify using a phone number or website you look up yourself, not one provided in the message. "
            "Report imposters at ReportFraud.ftc.gov."
        ),
    },
    {
        "doc_id": "ftc_phishing",
        "agency": "FTC",
        "title": "Phishing",
        "source_url": "https://consumer.ftc.gov/articles/how-recognize-and-avoid-phishing-scams",
        "topic": "phishing_spoofing",
        "text": (
            "Phishing scams use email, text, or social media messages that look like they come from a trusted source. "
            "Signs include unexpected requests for passwords, one-time codes, or account details, and links that "
            "urge you to act immediately. Do not click links or open attachments in unexpected messages. "
            "Go directly to the organization's site by typing the address yourself. "
            "If you shared information, change passwords and monitor accounts. Report phishing to ReportFraud.ftc.gov."
        ),
    },
    {
        "doc_id": "ssa_scams",
        "agency": "SSA",
        "title": "Social Security scams",
        "source_url": "https://www.ssa.gov/scam/",
        "topic": "government_impersonation",
        "text": (
            "The Social Security Administration will not threaten you with arrest or legal action, or demand immediate "
            "payment by gift card, wire transfer, or cryptocurrency. SSA generally will not call, text, or email to ask "
            "for your Social Security number or bank information unexpectedly. If you receive a suspicious message, "
            "hang up or ignore it, and contact Social Security through official channels you verify independently. "
            "Report Social Security scams to the Office of the Inspector General."
        ),
    },
    {
        "doc_id": "identitytheft_gov_steps",
        "agency": "IdentityTheft.gov",
        "title": "What to do right away",
        "source_url": "https://www.identitytheft.gov/Steps",
        "topic": "identity_theft_attempt",
        "text": (
            "If your personal information was exposed, IdentityTheft.gov recommends immediate steps such as placing a "
            "fraud alert or credit freeze, reviewing account statements, changing passwords, and creating a recovery plan. "
            "Use official recovery steps rather than links from the suspicious message. Keep records of what happened "
            "and which accounts may be affected. IdentityTheft.gov can help generate a personalized recovery plan."
        ),
    },
    {
        "doc_id": "ic3_psa_investment",
        "agency": "FBI IC3",
        "title": "Investment fraud caution",
        "source_url": "https://www.ic3.gov/",
        "topic": "investment_scam",
        "text": (
            "Investment scams often promise unusually high or guaranteed returns with little risk. "
            "They may pressure you to invest quickly through cryptocurrency or unfamiliar platforms and discourage "
            "independent verification. Be skeptical of unsolicited investment opportunities. Research firms through "
            "official regulators and report suspected cyber-enabled fraud to IC3.gov. Aggregate complaint statistics "
            "describe reported trends and do not prove that any one message is fraudulent."
        ),
    },
    {
        "doc_id": "ftc_online_shopping",
        "agency": "FTC",
        "title": "Online shopping scams",
        "source_url": "https://consumer.ftc.gov/articles/online-shopping",
        "topic": "phishing_spoofing",
        "text": (
            "Online shopping scams may advertise unrealistically low prices, use fake storefronts, or never deliver "
            "products. Check for secure checkout, clear seller identity, and independent reviews. Prefer known payment "
            "protections and avoid wiring money or paying with gift cards. If something goes wrong, contact your payment "
            "provider and report to ReportFraud.ftc.gov."
        ),
    },
    {
        "doc_id": "irs_impersonation",
        "agency": "IRS",
        "title": "Tax scams and IRS impersonation",
        "source_url": "https://www.irs.gov/newsroom/tax-scams-consumer-alerts",
        "topic": "government_impersonation",
        "text": (
            "The IRS will not call, text, or email to demand immediate payment with threats of arrest, deportation, "
            "or lawsuit, and will not ask for payment by gift card, prepaid debit card, wire transfer, or cryptocurrency. "
            "Threatening messages that claim you owe taxes and must pay right away are common scams. "
            "Verify any tax notice through IRS.gov or a number you look up yourself. Report IRS-related impersonation "
            "to the Treasury Inspector General for Tax Administration and to ReportFraud.ftc.gov."
        ),
    },
    {
        "doc_id": "ftc_romance_scams",
        "agency": "FTC",
        "title": "Romance scams",
        "source_url": "https://consumer.ftc.gov/articles/what-do-if-you-were-scammed-romance-scammer",
        "topic": "other_uncertain",
        "text": (
            "Romance scammers build trust online then ask for money for emergencies, travel, medical bills, or "
            "investments. They may refuse video calls, move conversations off a dating platform quickly, and pressure "
            "you to keep the relationship secret. Never send money, gift cards, or crypto to someone you have not "
            "met in person. If you sent money, stop contact, keep records, and report to ReportFraud.ftc.gov and local "
            "law enforcement. Banks and payment apps may also have fraud reporting options."
        ),
    },
    {
        "doc_id": "ftc_tech_support",
        "agency": "FTC",
        "title": "Tech support scams",
        "source_url": "https://consumer.ftc.gov/articles/how-spot-avoid-and-report-tech-support-scams",
        "topic": "phishing_spoofing",
        "text": (
            "Tech support scammers claim your computer or phone is infected and urge you to call a number, grant remote "
            "access, or buy software. Pop-ups, cold calls, and spoofed tech-brand messages are common. Legitimate "
            "companies do not cold-call to fix devices this way. Do not give remote control of your device or pay for "
            "unexpected support. Hang up, close the browser, and contact the company only through official support "
            "channels you verify yourself. Report tech support scams to ReportFraud.ftc.gov."
        ),
    },
    {
        "doc_id": "ftc_job_scams",
        "agency": "FTC",
        "title": "Job and employment scams",
        "source_url": "https://consumer.ftc.gov/articles/job-scams",
        "topic": "other_uncertain",
        "text": (
            "Job scams may offer high pay for little work, ask you to pay for training or equipment up front, or send "
            "fake checks and ask you to wire money back. They often recruit over text or social media and rush you to "
            "share bank details or Social Security numbers. Research employers independently, never pay to get a job, "
            "and be wary of checks that ask you to refund money. Report job scams to ReportFraud.ftc.gov."
        ),
    },
    {
        "doc_id": "ftc_gift_cards",
        "agency": "FTC",
        "title": "How to avoid gift card scams",
        "source_url": "https://consumer.ftc.gov/articles/how-avoid-gift-card-scams",
        "topic": "other_uncertain",
        "text": (
            "A request to pay with gift cards is a major red flag. Government agencies, utilities, and reputable "
            "businesses do not demand gift-card payment for taxes, bail, utilities, or debts. Scammers tell you to "
            "buy cards, scratch the codes, and read them over the phone or text. Once codes are shared, the money is "
            "usually gone. Refuse gift-card payment demands, keep receipts if you already bought cards, and report "
            "promptly to the card issuer and ReportFraud.ftc.gov."
        ),
    },
    {
        "doc_id": "cfpb_bank_impersonation",
        "agency": "CFPB",
        "title": "Fraudulent bank and account messages",
        "source_url": "https://www.consumerfinance.gov/consumer-tools/fraud/",
        "topic": "banking_impersonation",
        "text": (
            "Messages that claim to be from your bank may ask you to confirm a transfer, unlock an account, or share "
            "a one-time passcode. Banks generally will not ask you to reveal full passwords or one-time codes to "
            "complete a surprise transaction. If you get an unexpected alert, contact your bank using the number on "
            "your card or the official app, not a number in the text. Monitor accounts, enable alerts you control, "
            "and report unauthorized transactions quickly under your bank's dispute process."
        ),
    },
    {
        "doc_id": "ftc_credit_freeze",
        "agency": "FTC",
        "title": "Credit freezes and fraud alerts",
        "source_url": "https://consumer.ftc.gov/articles/credit-freeze-faqs",
        "topic": "identity_theft_attempt",
        "text": (
            "A credit freeze restricts access to your credit report and can make it harder for identity thieves to "
            "open new accounts in your name. You can place free freezes with Equifax, Experian, and TransUnion. "
            "A fraud alert asks lenders to take extra steps to verify your identity. These steps do not stop all "
            "fraud, so also review statements and IdentityTheft.gov recovery guidance. Place freezes through the "
            "bureaus' official sites, not through links in a suspicious message."
        ),
    },
    {
        "doc_id": "fcc_robocall_text",
        "agency": "FCC",
        "title": "Unwanted calls and texts",
        "source_url": "https://www.fcc.gov/consumers/guides/stop-unwanted-robocalls-and-texts",
        "topic": "phishing_spoofing",
        "text": (
            "Unwanted robocalls and scam texts often spoof familiar numbers and push you to call back or click a link. "
            "Do not engage with unexpected messages that demand money or personal data. Use carrier spam-blocking "
            "tools, register numbers on the National Do Not Call Registry for telemarketing, and report illegal "
            "robocalls and texts to the FCC. Blocking and reporting help, but verify any real account issue through "
            "official apps or websites you already trust."
        ),
    },
    {
        "doc_id": "cisa_phishing_guidance",
        "agency": "CISA",
        "title": "Avoiding social engineering and phishing",
        "source_url": "https://www.cisa.gov/news-events/news/avoiding-social-engineering-and-phishing-attacks",
        "topic": "phishing_spoofing",
        "text": (
            "Social engineering tricks people into revealing credentials or installing malware. Common cues include "
            "urgent deadlines, unusual sender addresses, unexpected attachments, and requests that bypass normal "
            "procedures. Pause before clicking, hover over links carefully, and verify requests through a separate "
            "channel. Use multifactor authentication and unique passwords. Report suspicious messages to your "
            "organization's security team when applicable, and to consumer reporting sites for personal accounts."
        ),
    },
    {
        "doc_id": "ftc_crypto_scams",
        "agency": "FTC",
        "title": "Cryptocurrency investment and payment scams",
        "source_url": "https://consumer.ftc.gov/articles/what-know-about-cryptocurrency-and-scams",
        "topic": "investment_scam",
        "text": (
            "Crypto-related scams include fake investment platforms, pig-butchering romance-investment schemes, and "
            "demands to pay bills or bail in cryptocurrency. Scammers may show fabricated account balances and urge "
            "more deposits. Crypto transfers are hard to reverse. Research platforms independently, be skeptical of "
            "guaranteed returns, and never pay government or company debts in crypto. Report crypto fraud to "
            "ReportFraud.ftc.gov and IC3.gov."
        ),
    },
    {
        "doc_id": "medicare_scams",
        "agency": "CMS",
        "title": "Medicare fraud and scams",
        "source_url": "https://www.medicare.gov/basics/reporting-medicare-fraud-and-abuse",
        "topic": "government_impersonation",
        "text": (
            "Medicare will not call you to ask for your Medicare number or bank information to issue a new card or "
            "update records without a clear, expected reason. Scammers may offer free genetic tests, equipment, or "
            "money and then steal beneficiary information. Protect your Medicare number like a credit card. Confirm "
            "requests through Medicare.gov or 1-800-MEDICARE using numbers you look up yourself. Report suspected "
            "Medicare fraud through official CMS reporting channels."
        ),
    },
    {
        "doc_id": "usps_package_scams",
        "agency": "USPIS",
        "title": "Package and delivery text scams",
        "source_url": "https://www.uspis.gov/news/consumer-awareness/",
        "topic": "phishing_spoofing",
        "text": (
            "Fake delivery texts claim a package could not be delivered and ask you to pay a small fee or confirm "
            "an address through a link. These links often lead to phishing sites that steal card details or login "
            "credentials. Track packages only through the carrier's official app or site using a tracking number "
            "you already have. Do not click unexpected delivery links. Report suspicious postal-related scams to "
            "the U.S. Postal Inspection Service."
        ),
    },
    {
        "doc_id": "ftc_family_emergency",
        "agency": "FTC",
        "title": "Family emergency and grandparent scams",
        "source_url": "https://consumer.ftc.gov/articles/family-emergency-scams",
        "topic": "other_uncertain",
        "text": (
            "Family emergency scams claim a relative is in jail, in an accident, or stranded and needs money now. "
            "Callers may pretend to be a lawyer, police officer, or the relative and insist on secrecy. Verify the "
            "story by calling the relative or another family member using a known phone number. Do not wire money "
            "or buy gift cards based on a sudden crisis call. Report these scams to ReportFraud.ftc.gov."
        ),
    },
    {
        "doc_id": "sec_investment_red_flags",
        "agency": "SEC",
        "title": "Investment fraud red flags",
        "source_url": "https://www.investor.gov/protect-your-investments/fraud/how-avoid-fraud",
        "topic": "investment_scam",
        "text": (
            "Investment fraud often features promises of high returns with little or no risk, pressure to act "
            "immediately, unregistered sellers, and complex products you cannot explain. Check registration of "
            "advisers and brokers through Investor.gov tools before sending money. Be cautious of tips from social "
            "media influencers and unsolicited investment texts. If something feels wrong, stop and verify. Report "
            "suspected securities fraud to the SEC and to IC3 for online schemes."
        ),
    },
    {
        "doc_id": "ftc_account_takeover",
        "agency": "FTC",
        "title": "Account takeover and SIM swap risks",
        "source_url": "https://consumer.ftc.gov/articles/protecting-personal-information-guide-business",
        "topic": "identity_theft_attempt",
        "text": (
            "Account takeover happens when someone gains access to your email, social media, or financial accounts. "
            "SIM swap and phone-port scams can intercept one-time codes sent by text. Use app-based multifactor "
            "authentication when available, unique passwords, and account recovery contacts you control. If your "
            "phone suddenly loses service, contact your carrier immediately. Change passwords from a trusted device "
            "and review recent account activity for unauthorized changes."
        ),
    },
    {
        "doc_id": "ftc_debt_collection",
        "agency": "FTC",
        "title": "Fake debt collection scams",
        "source_url": "https://consumer.ftc.gov/articles/debt-collection-faqs",
        "topic": "other_uncertain",
        "text": (
            "Fake debt collectors threaten arrest, lawsuit, or wage garnishment unless you pay immediately, often "
            "by wire or gift card. Real collectors must follow rules and should provide written validation of a debt. "
            "Do not pay based on a threatening call alone. Ask for the debt details in writing and verify with "
            "creditors you recognize. Report abusive or fraudulent collection attempts to the CFPB and "
            "ReportFraud.ftc.gov."
        ),
    },
    {
        "doc_id": "ftc_housing_rental",
        "agency": "FTC",
        "title": "Rental and housing scams",
        "source_url": "https://consumer.ftc.gov/articles/rental-listing-scams",
        "topic": "phishing_spoofing",
        "text": (
            "Rental scams advertise places that are not available or not owned by the person listing them. Scammers "
            "may ask for a deposit by wire, gift card, or app payment before you see the unit or sign a lease. "
            "Visit properties in person when possible, research landlords, and never send money for a rental you "
            "have not verified. If a deal seems too good to be true, walk away and report the listing and payment "
            "request to ReportFraud.ftc.gov."
        ),
    },
    {
        "doc_id": "onguard_online_passwords",
        "agency": "FTC",
        "title": "Strong passwords and account security",
        "source_url": "https://consumer.ftc.gov/articles/how-create-strong-password",
        "topic": "identity_theft_attempt",
        "text": (
            "Strong, unique passwords and multifactor authentication reduce the damage from phishing and data "
            "breaches. Do not reuse passwords across banks, email, and shopping sites. Prefer a password manager "
            "and avoid sharing one-time codes even if a message claims to be from customer support. After a "
            "suspicious message or possible compromise, change passwords starting with email and financial accounts, "
            "then review recovery phone numbers and trusted devices."
        ),
    },
]


def main() -> None:
    settings = get_settings()
    store = get_store()
    inventory = []
    today = date.today().isoformat()
    knowledge_key = settings.knowledge_key.rstrip("/")
    provenance_key = settings.provenance_key.rstrip("/")

    for doc in DOCS:
        relative = f"{knowledge_key}/{doc['doc_id']}.json"
        payload = {
            **doc,
            "retrieved_date": today,
            "license_note": "Public US government educational guidance for RAG citations.",
            "immutable_raw": False,
            "synthetic": False,
            "storage": "s3",
        }
        uri = store.write_json(relative, payload)
        inventory.append(
            {
                "doc_id": doc["doc_id"],
                "agency": doc["agency"],
                "title": doc["title"],
                "source_url": doc["source_url"],
                "topic": doc["topic"],
                "retrieved_date": today,
                "s3_uri": uri,
                "s3_key": relative,
            }
        )
        print("wrote", uri)

    prov_relative = f"{provenance_key}/knowledge_inventory.json"
    prov_uri = store.write_json(
        prov_relative,
        {
            "retrieved_date": today,
            "storage": "s3",
            "s3_prefix": store.s3_uri,
            "documents": inventory,
        },
    )
    print("wrote", prov_uri)
    print(f"seeded {len(DOCS)} documents to S3")


if __name__ == "__main__":
    main()
