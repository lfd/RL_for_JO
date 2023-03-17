SELECT * FROM title AS t, role_type AS rt, cast_info AS ci, char_name AS chn, movie_companies AS mc WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND mc.note IS NOT NULL AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND rt.role = 'actress' AND t.production_year BETWEEN 2005 AND 2009 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND rt.id = ci.role_id AND ci.role_id = rt.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;