SELECT * FROM comp_cast_type AS cct2, movie_companies AS mc, complete_cast AS cc, comp_cast_type AS cct1 WHERE cct1.kind = 'crew' AND cct2.kind != 'complete+verified' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;