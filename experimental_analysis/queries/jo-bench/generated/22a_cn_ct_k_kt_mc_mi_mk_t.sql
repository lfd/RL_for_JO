SELECT * FROM company_name AS cn, kind_type AS kt, movie_keyword AS mk, company_type AS ct, movie_companies AS mc, title AS t, movie_info AS mi, keyword AS k WHERE cn.country_code != '[us]' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND kt.kind IN ('movie', 'episode') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi.info IN ('Germany', 'German', 'USA', 'American') AND t.production_year > 2008 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id;