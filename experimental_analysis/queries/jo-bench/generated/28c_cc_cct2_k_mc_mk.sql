SELECT * FROM comp_cast_type AS cct2, keyword AS k, movie_keyword AS mk, complete_cast AS cc, movie_companies AS mc WHERE cct2.kind = 'complete' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;