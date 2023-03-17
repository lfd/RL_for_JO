SELECT * FROM comp_cast_type AS cct2, complete_cast AS cc, title AS t, comp_cast_type AS cct1, cast_info AS ci, aka_name AS an, char_name AS chn WHERE cct1.kind = 'cast' AND cct2.kind = 'complete+verified' AND ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND t.production_year BETWEEN 2000 AND 2010 AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;