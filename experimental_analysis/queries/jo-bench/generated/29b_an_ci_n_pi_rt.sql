SELECT * FROM name AS n, person_info AS pi, role_type AS rt, cast_info AS ci, aka_name AS an WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%An%' AND rt.role = 'actress' AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id;