SELECT * FROM cast_info AS ci, aka_name AS an, role_type AS rt, title AS t, movie_companies AS mc, name AS n, char_name AS chn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND mc.note IS NOT NULL AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND n.gender = 'f' AND n.name LIKE '%Ang%' AND rt.role = 'actress' AND t.production_year BETWEEN 2005 AND 2015 AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND an.person_id = n.id AND n.id = an.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;