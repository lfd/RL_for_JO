SELECT * FROM title AS t, cast_info AS ci, char_name AS chn, role_type AS rt WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND rt.role = 'actress' AND t.production_year BETWEEN 2005 AND 2009 AND t.id = ci.movie_id AND ci.movie_id = t.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;