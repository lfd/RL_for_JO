SELECT * FROM cast_info AS ci, role_type AS rt WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND rt.role = 'actress' AND ci.role_id = rt.id AND rt.id = ci.role_id;