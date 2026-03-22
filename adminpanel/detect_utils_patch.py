"""
PATCH FILE — only two functions changed in detect_utils.py
Replace save_alert() and send_alert_email() with these versions.
Everything else in your detect_utils.py stays exactly the same.
"""

    @staticmethod
    def save_alert(frame, label, confidence, severity=None, extra_info=None):
        if extra_info is None:
            extra_info = {}

        # Convert frame to image for storage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        img_io = io.BytesIO()
        pil_img.save(img_io, format='JPEG', quality=70)

        safe_label = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in label]).strip().replace(' ', '_')
        img_content = ContentFile(img_io.getvalue(), name=f"alert_{safe_label}.jpg")

        if severity is None:
            severity = 'medium'
            label_lower = label.lower()
            if 'fire' in label_lower or 'default' in label_lower: severity = 'high'
            elif 'class b' in label_lower:                         severity = 'critical'
            elif 'cylinder' in label_lower or 'gas' in label_lower: severity = 'critical'
            elif 'smoke' in label_lower:                           severity = 'high'
            elif 'burning' in label_lower:                         severity = 'high'

        display_label = label
        if label.lower() == 'default':
            display_label = 'Fire/Burning Detected'

        alert = Alert.objects.create(
            alert_type=display_label,
            confidence=confidence,
            severity=severity,
            snapshot=img_content
        )

        import threading

        def send_notifications_bg(alert_instance, info, label_text):
            try:
                FireDetector.send_alert_email(alert_instance, info)
            except Exception as e:
                print(f"Error sending email alert: {e}")

            try:
                from .views import send_push_notification

                f_class   = info.get('fire_class', 'A')
                materials = info.get('burning_materials', [])

                # If no materials detected default to paper (most likely in a classroom demo)
                mat_str = ", ".join(materials) if materials else "Paper (Ordinary Combustible)"

                # Correct extinguisher per fire class
                exting_map = {
                    'A': 'Water / Foam / ABC Dry Powder',
                    'B': 'Foam / CO2 / Dry Powder',
                    'C': 'CO2 / Dry Powder (DO NOT USE WATER)',
                    'K': 'Wet Chemical',
                }
                exting = exting_map.get(f_class, 'Water / Foam / ABC Dry Powder')

                # Class label shown in notification title
                class_label = f"Class {f_class} Fire" if f_class else "Fire"

                title = f"🔥 FIRE ALERT: {class_label} Detected"
                time_now = alert_instance.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                body = (
                    f"Fire Class  : {f_class} — Ordinary Combustible\n"
                    f"Material    : {mat_str}\n"
                    f"Extinguisher: {exting}\n"
                    f"Severity    : {severity.upper()}\n"
                    f"Location    : {alert_instance.location}\n"
                    f"Time        : {time_now}"
                )

                send_push_notification(title, body, {'alert_id': alert_instance.id})

            except Exception as e:
                print(f"Error sending push notification: {e}")

        threading.Thread(
            target=send_notifications_bg,
            args=(alert, extra_info, display_label)
        ).start()

    @staticmethod
    def send_alert_email(alert, extra_info=None):
        if extra_info is None:
            extra_info = {}

        from django.core.mail import EmailMultiAlternatives
        from django.conf import settings
        from .models import AlertRecipient

        is_critical = alert.severity in ['high', 'critical']
        if is_critical:
            recipients = AlertRecipient.objects.filter(is_active=True)
        else:
            recipients = AlertRecipient.objects.filter(is_active=True, receive_critical_only=False)

        if not recipients.exists():
            return

        recipient_list = [r.email for r in recipients]

        f_class   = extra_info.get('fire_class', 'A')
        materials = extra_info.get('burning_materials', [])
        mat_str   = ", ".join(materials) if materials else "Paper (Ordinary Combustible)"

        exting_map = {
            'A': 'Water, Foam, or ABC Dry Powder',
            'B': 'Foam, CO2, or Dry Powder',
            'C': 'CO2 or Dry Powder — DO NOT USE WATER',
            'K': 'Wet Chemical',
        }
        exting = exting_map.get(f_class, 'Water, Foam, or ABC Dry Powder')

        class_label = f"Class {f_class} — Ordinary Combustible" if f_class == 'A' else f"Class {f_class}"

        # Email subject
        icon    = "🔥" if alert.severity in ['high', 'critical'] else "☁️"
        subject = f"{icon} FIRE GUARD: {class_label} Fire Detected [{alert.severity.upper()}]"

        # Plain text fallback
        text_body = f"""
FIRE GUARD SECURITY ALERT
=========================

A Class {f_class} fire has been detected by the AI Surveillance System.

DETAILS:
  Alert Type   : {alert.alert_type}
  Fire Class   : {f_class} — Ordinary Combustible (Paper / Wood / Fabric)
  Material     : {mat_str}
  Extinguisher : {exting}
  Severity     : {alert.severity.upper()}
  Location     : {alert.location}
  Time         : {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

RECOMMENDED ACTION:
  Use {exting} to extinguish.
  Evacuate the area if the fire cannot be controlled immediately.

--
Automated Alert — Fire Guard AI System
        """

        # Colour based on severity
        color = "#e11d48" if alert.severity == 'critical' else "#f97316" if alert.severity == 'high' else "#6b7280"

        # HTML email
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    body        {{ font-family: 'Helvetica Neue', Arial, sans-serif; color: #333; line-height: 1.6; margin: 0; padding: 0; }}
    .wrap       {{ max-width: 600px; margin: 30px auto; border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; }}
    .header     {{ background: {color}; color: #fff; padding: 24px 20px; text-align: center; }}
    .header h1  {{ margin: 0; font-size: 22px; letter-spacing: 0.5px; }}
    .header p   {{ margin: 6px 0 0; font-size: 13px; opacity: 0.85; }}
    .body       {{ padding: 28px; background: #ffffff; }}
    .class-badge {{
        display: inline-block;
        background: {color};
        color: #fff;
        font-size: 18px;
        font-weight: 700;
        padding: 8px 22px;
        border-radius: 999px;
        margin-bottom: 18px;
    }}
    .info-box   {{ background: #fff7ed; border-left: 4px solid {color}; border-radius: 6px; padding: 16px; margin: 16px 0; }}
    table       {{ width: 100%; border-collapse: collapse; }}
    td          {{ padding: 9px 4px; border-bottom: 1px solid #f3f4f6; font-size: 14px; }}
    .lbl        {{ font-weight: 600; color: #555; width: 38%; }}
    .val        {{ color: #111; }}
    .val.hi     {{ color: {color}; font-weight: 700; }}
    .val.green  {{ color: #16a34a; font-weight: 700; }}
    .action-box {{ background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;
                   padding: 14px 18px; margin-top: 20px; font-size: 14px; }}
    .footer     {{ background: #f9fafb; padding: 14px; text-align: center;
                   font-size: 11px; color: #9ca3af; }}
  </style>
</head>
<body>
  <div class="wrap">

    <div class="header">
      <h1>🔥 FIRE GUARD — FIRE ALERT</h1>
      <p>AI Surveillance System — Automatic Notification</p>
    </div>

    <div class="body">
      <div class="class-badge">CLASS {f_class} — Ordinary Combustible</div>

      <p style="margin-top:0; font-size:15px;">
        A <strong>Class {f_class} fire</strong> involving <strong>{mat_str}</strong>
        has been detected by the FireGuard AI system.
      </p>

      <div class="info-box">
        <table>
          <tr>
            <td class="lbl">Alert Type</td>
            <td class="val"><strong>{alert.alert_type}</strong></td>
          </tr>
          <tr>
            <td class="lbl">Fire Class</td>
            <td class="val hi">Class {f_class} — Ordinary Combustible</td>
          </tr>
          <tr>
            <td class="lbl">Burning Material</td>
            <td class="val hi">{mat_str}</td>
          </tr>
          <tr>
            <td class="lbl">Extinguisher</td>
            <td class="val green">✅ {exting}</td>
          </tr>
          <tr>
            <td class="lbl">Severity</td>
            <td class="val hi">{alert.severity.upper()}</td>
          </tr>
          <tr>
            <td class="lbl">Location</td>
            <td class="val">{alert.location}</td>
          </tr>
          <tr>
            <td class="lbl">Time</td>
            <td class="val">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
          </tr>
        </table>
      </div>

      <div class="action-box">
        <strong>⚠️ Recommended Action</strong><br>
        Use <strong>{exting}</strong> to extinguish the fire.<br>
        If the fire cannot be controlled immediately, <strong>evacuate the area</strong>
        and contact emergency services.
      </div>

      <p style="margin-top:18px; font-size:13px; color:#6b7280; text-align:center;">
        A snapshot from the surveillance feed is attached below.
      </p>
    </div>

    <div class="footer">
      &copy; 2025 Fire Guard AI System &nbsp;|&nbsp; Automated Message — Do not reply
    </div>

  </div>
</body>
</html>
        """

        email = EmailMultiAlternatives(
            subject,
            text_body,
            settings.DEFAULT_FROM_EMAIL,
            [],
            recipient_list,
        )
        email.attach_alternative(html_body, "text/html")

        # Attach snapshot
        if alert.snapshot:
            try:
                alert.snapshot.open('rb')
                email.attach(alert.snapshot.name, alert.snapshot.read(), 'image/jpeg')
                alert.snapshot.close()
            except Exception as e:
                print(f"Could not attach image: {e}")

        print(f"Sending email to {len(recipient_list)} recipients...")
        email.send(fail_silently=False)
        print("Email sent successfully.")